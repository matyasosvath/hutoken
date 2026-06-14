#include "hutoken/core.h"

#include "Python.h"
#ifdef USE_FOMA
#include "fomalib.h"
#endif

#include <assert.h>
#include <regex.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hutoken/arena.h"
#include "hutoken/hashmap.h"
#include "hutoken/helper.h"
#include "hutoken/parser.h"
#include "hutoken/pretokenizer.h"
#include "hutoken/queue.h"
#include "hutoken/taskqueue.h"
#include "hutoken/vector.h"

static const size_t FIXED_ARENA_SIZE = (size_t)16 * 1024 * 1024;
static const size_t BPE_ARENA_MULTIPLIER = 64;

struct TokenNode {
    int prev;
    int next;
};

int hex_token_length(const char* ptr) {
    if (ptr[0] == '<' && ptr[1] == '0' && (ptr[2] == 'x' || ptr[2] == 'X')) {
        const char* p = ptr + 3;
        while ((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f') ||
               (*p >= 'A' && *p <= 'F')) {
            p++;
        }
        if (*p == '>') {
            return (p - ptr) + 1;
        }
    }
    return -1;
}

int next_token_length(const char* ptr) {
    int hex_len = hex_token_length(ptr);
    if (hex_len > 0) {
        return hex_len;
    }
    return utf8_char_length((const unsigned char*)ptr);
}

static int get_pair_rank_from_strings(const struct HashMap* vocab,
                                      const struct Boundary token_boundaries[],
                                      const int left_idx,
                                      const int right_idx);

static int get_pair_rank_from_ids(const struct HashMap* merges_map,
                                  const int left_id,
                                  const int right_id);

static void* allocate_buffer(struct Arena* arena, bool use_arena,
                             size_t size) {
    if (!use_arena) {
        return malloc(size);
    }
    return arena_alloc(arena, size);
}

static void free_buffer(void* ptr, bool use_arena) {
    if (!use_arena && ptr) {
        free(ptr);
    }
}

static enum MinPQError init_pq_for_bpe(struct Arena* arena,
                                       struct MinPQ* pq,
                                       const size_t capacity,
                                       bool use_arena) {
    return use_arena ? min_pq_init_arena(arena, pq, capacity)
                     : min_pq_init(pq, capacity);
}

static enum MinPQError push_pq_for_bpe(struct Arena* arena,
                                       struct MinPQ* pq,
                                       const struct MergeCandidate candidate,
                                       bool use_arena) {
    return use_arena ? min_pq_push_arena(arena, pq, candidate)
                     : min_pq_push(pq, candidate);
}

static void bpe_encode_string(struct Arena* arena,
                              bool use_arena,
                              bool use_optimized,
                              struct HashMap* vocab,
                              struct Boundary token_boundaries[],
                              int tokens[],
                              int* token_num) {
    if (use_optimized) {
        struct MinPQ pq;
        if (init_pq_for_bpe(arena, &pq, *token_num, use_arena) !=
            MIN_PQ_SUCCESS) {
            log_debug("Failed to initialize priority queue.");
            return;
        }

        struct TokenNode* nodes =
            allocate_buffer(arena, use_arena, *token_num * sizeof(struct TokenNode));
        bool* consumed = allocate_buffer(arena, use_arena,
                                        *token_num * sizeof(bool));
        if (!nodes || !consumed) {
            log_debug("Failed to allocate memory for token nodes.");
            free_buffer(nodes, use_arena);
            free_buffer(consumed, use_arena);
            if (!use_arena) {
                min_pq_release(&pq);
            }
            return;
        }

        memset(consumed, 0, *token_num * sizeof(bool));

        for (int i = 0; i < *token_num; ++i) {
            nodes[i].prev = i - 1;
            nodes[i].next = i + 1;
        }
        nodes[*token_num - 1].next = -1;

        for (int i = 0; i < *token_num - 1; ++i) {
            const int rank = get_pair_rank_from_strings(vocab, token_boundaries,
                                                       i, i + 1);
            if (rank != -1) {
                const struct MergeCandidate candidate = {
                    .rank = rank, .left_idx = i, .right_idx = i + 1};

                if (push_pq_for_bpe(arena, &pq, candidate, use_arena) !=
                    MIN_PQ_SUCCESS) {
                    log_debug("Failed to push to queue.");
                    free_buffer(nodes, use_arena);
                    free_buffer(consumed, use_arena);
                    if (!use_arena) {
                        min_pq_release(&pq);
                    }
                    return;
                }
            }
        }

        while (!min_pq_is_empty(&pq)) {
            struct MergeCandidate best_pair = {0};
            (void)min_pq_pop(&pq, &best_pair);

            const int left_idx = best_pair.left_idx;
            const int right_idx = best_pair.right_idx;

            if (consumed[left_idx] || consumed[right_idx]) {
                continue;
            }

            if (nodes[left_idx].next != right_idx) {
                continue;
            }

            const int current_rank = get_pair_rank_from_strings(
                vocab, token_boundaries, left_idx, right_idx);
            if (best_pair.rank != current_rank) {
                continue;
            }

            token_boundaries[left_idx].end = token_boundaries[right_idx].end;
            consumed[right_idx] = true;

            const int prev_idx = nodes[left_idx].prev;
            const int next_idx = nodes[right_idx].next;
            nodes[left_idx].next = next_idx;
            if (next_idx != -1) {
                nodes[next_idx].prev = left_idx;
            }

            if (prev_idx != -1) {
                const int rank = get_pair_rank_from_strings(
                    vocab, token_boundaries, prev_idx, left_idx);
                if (rank != -1) {
                    if (push_pq_for_bpe(
                            arena, &pq,
                            (struct MergeCandidate){.rank = rank,
                                                    .left_idx = prev_idx,
                                                    .right_idx = left_idx},
                            use_arena) != MIN_PQ_SUCCESS) {
                        log_debug("Failed to push to queue.");
                        free_buffer(nodes, use_arena);
                        free_buffer(consumed, use_arena);
                        if (!use_arena) {
                            min_pq_release(&pq);
                        }
                        return;
                    }
                }
            }

            if (next_idx != -1) {
                const int rank = get_pair_rank_from_strings(
                    vocab, token_boundaries, left_idx, next_idx);
                if (rank != -1) {
                    if (push_pq_for_bpe(
                            arena, &pq,
                            (struct MergeCandidate){.rank = rank,
                                                    .left_idx = left_idx,
                                                    .right_idx = next_idx},
                            use_arena) != MIN_PQ_SUCCESS) {
                        log_debug("Failed to push to queue.");
                        free_buffer(nodes, use_arena);
                        free_buffer(consumed, use_arena);
                        if (!use_arena) {
                            min_pq_release(&pq);
                        }
                        return;
                    }
                }
            }
        }

        if (!use_arena) {
            min_pq_release(&pq);
        }

        struct Boundary* final_boundaries =
            allocate_buffer(arena, use_arena, *token_num * sizeof(struct Boundary));
        if (!final_boundaries) {
            log_debug("Failed to allocate memory for final boundaries.");
            free_buffer(nodes, use_arena);
            free_buffer(consumed, use_arena);
            return;
        }

        int final_token_count = 0;
        for (int i = 0; i < *token_num; ++i) {
            if (!consumed[i]) {
                final_boundaries[final_token_count++] = token_boundaries[i];
            }
        }

        memcpy(token_boundaries, final_boundaries,
               final_token_count * sizeof(struct Boundary));
        *token_num = final_token_count;

        if (!use_arena) {
            free_buffer(final_boundaries, use_arena);
        }
        free_buffer(nodes, use_arena);
        free_buffer(consumed, use_arena);
    } else {
        bool* consumed = allocate_buffer(arena, use_arena,
                                        *token_num * sizeof(bool));
        if (!consumed) {
            log_debug("Failed to allocate memory for token consumed flags.");
            return;
        }

        memset(consumed, 0, *token_num * sizeof(bool));

        while (true) {
            int best_rank = -1;
            int best_left = -1;
            int best_right = -1;
            int prev_active = -1;

            for (int i = 0; i < *token_num; ++i) {
                if (consumed[i]) {
                    continue;
                }
                if (prev_active != -1) {
                    const int rank = get_pair_rank_from_strings(
                        vocab, token_boundaries, prev_active, i);
                    if (rank != -1 && rank > best_rank) {
                        best_rank = rank;
                        best_left = prev_active;
                        best_right = i;
                    }
                }
                prev_active = i;
            }

            if (best_left == -1) {
                break;
            }

            token_boundaries[best_left].end = token_boundaries[best_right].end;
            consumed[best_right] = true;
        }

        struct Boundary* final_boundaries =
            allocate_buffer(arena, use_arena, *token_num * sizeof(struct Boundary));
        if (!final_boundaries) {
            log_debug("Failed to allocate memory for final boundaries.");
            free_buffer(consumed, use_arena);
            return;
        }

        int final_token_count = 0;
        for (int i = 0; i < *token_num; ++i) {
            if (!consumed[i]) {
                final_boundaries[final_token_count++] = token_boundaries[i];
            }
        }

        memcpy(token_boundaries, final_boundaries,
               final_token_count * sizeof(struct Boundary));
        *token_num = final_token_count;

        if (!use_arena) {
            free_buffer(final_boundaries, use_arena);
        }
        free_buffer(consumed, use_arena);
    }

    for (int i = 0; i < *token_num; ++i) {
        const char* start = token_boundaries[i].start;
        const char* end = token_boundaries[i].end;
        const ptrdiff_t len = (end - start) + 1;

        char token_str[len + 1];
        memcpy(token_str, start, len);
        token_str[len] = '\0';

        const struct Token* found_token =
            hashmap_get(vocab, &(struct Token){.key = token_str});
        tokens[i] = (found_token != NULL) ? found_token->value : -1;
    }
}

static void bpe_encode_ids(struct Arena* arena,
                           bool use_arena,
                           bool use_optimized,
                           struct HashMap* merges_map,
                           int tokens[],
                           int* token_num) {
    if (use_optimized) {
        struct MinPQ pq;
        if (init_pq_for_bpe(arena, &pq, *token_num, use_arena) !=
            MIN_PQ_SUCCESS) {
            log_debug("Failed to initialize priority queue.");
            return;
        }

        struct TokenNode* nodes =
            allocate_buffer(arena, use_arena, *token_num * sizeof(struct TokenNode));
        bool* consumed = allocate_buffer(arena, use_arena,
                                        *token_num * sizeof(bool));
        if (!nodes || !consumed) {
            log_debug("Failed to allocate memory for token nodes.");
            free_buffer(nodes, use_arena);
            free_buffer(consumed, use_arena);
            if (!use_arena) {
                min_pq_release(&pq);
            }
            return;
        }

        memset(consumed, 0, *token_num * sizeof(bool));

        for (int i = 0; i < *token_num; ++i) {
            nodes[i].prev = i - 1;
            nodes[i].next = i + 1;
        }
        nodes[*token_num - 1].next = -1;

        for (int i = 0; i < *token_num - 1; ++i) {
            const int rank = get_pair_rank_from_ids(merges_map, tokens[i],
                                                    tokens[i + 1]);
            if (rank != -1) {
                const struct MergeCandidate candidate = {
                    .rank = rank, .left_idx = i, .right_idx = i + 1};

                if (push_pq_for_bpe(arena, &pq, candidate, use_arena) !=
                    MIN_PQ_SUCCESS) {
                    log_debug("Failed to push to queue.");
                    free_buffer(nodes, use_arena);
                    free_buffer(consumed, use_arena);
                    if (!use_arena) {
                        min_pq_release(&pq);
                    }
                    return;
                }
            }
        }

        while (!min_pq_is_empty(&pq)) {
            struct MergeCandidate best_pair = {0};
            (void)min_pq_pop(&pq, &best_pair);

            const int left_idx = best_pair.left_idx;
            const int right_idx = best_pair.right_idx;

            if (consumed[left_idx] || consumed[right_idx]) {
                continue;
            }

            if (nodes[left_idx].next != right_idx) {
                continue;
            }

            const int current_rank = get_pair_rank_from_ids(
                merges_map, tokens[left_idx], tokens[right_idx]);
            if (best_pair.rank != current_rank) {
                continue;
            }

            struct MergeRule key = {.left_id = tokens[left_idx],
                                    .right_id = tokens[right_idx]};
            const struct MergeRule* rule = hashmap_get(merges_map, &key);
            if (!rule) {
                continue;
            }

            tokens[left_idx] = rule->merge_id;
            consumed[right_idx] = true;

            const int prev_idx = nodes[left_idx].prev;
            const int next_idx = nodes[right_idx].next;
            nodes[left_idx].next = next_idx;
            if (next_idx != -1) {
                nodes[next_idx].prev = left_idx;
            }

            if (prev_idx != -1) {
                const int rank = get_pair_rank_from_ids(
                    merges_map, tokens[prev_idx], tokens[left_idx]);
                if (rank != -1) {
                    if (push_pq_for_bpe(
                            arena, &pq,
                            (struct MergeCandidate){.rank = rank,
                                                    .left_idx = prev_idx,
                                                    .right_idx = left_idx},
                            use_arena) != MIN_PQ_SUCCESS) {
                        log_debug("Failed to push to queue.");
                        free_buffer(nodes, use_arena);
                        free_buffer(consumed, use_arena);
                        if (!use_arena) {
                            min_pq_release(&pq);
                        }
                        return;
                    }
                }
            }

            if (next_idx != -1) {
                const int rank = get_pair_rank_from_ids(
                    merges_map, tokens[left_idx], tokens[next_idx]);
                if (rank != -1) {
                    if (push_pq_for_bpe(
                            arena, &pq,
                            (struct MergeCandidate){.rank = rank,
                                                    .left_idx = left_idx,
                                                    .right_idx = next_idx},
                            use_arena) != MIN_PQ_SUCCESS) {
                        log_debug("Failed to push to queue.");
                        free_buffer(nodes, use_arena);
                        free_buffer(consumed, use_arena);
                        if (!use_arena) {
                            min_pq_release(&pq);
                        }
                        return;
                    }
                }
            }
        }

        if (!use_arena) {
            min_pq_release(&pq);
        }

        int final_token_count = 0;
        for (int i = 0; i < *token_num; ++i) {
            if (!consumed[i]) {
                tokens[final_token_count++] = tokens[i];
            }
        }

        *token_num = final_token_count;
        free_buffer(nodes, use_arena);
        free_buffer(consumed, use_arena);
    } else {
        bool* consumed = allocate_buffer(arena, use_arena,
                                        *token_num * sizeof(bool));
        if (!consumed) {
            log_debug("Failed to allocate memory for token consumed flags.");
            return;
        }

        memset(consumed, 0, *token_num * sizeof(bool));

        while (true) {
            int best_rank = -1;
            int best_left = -1;
            int best_right = -1;
            int prev_active = -1;

            for (int i = 0; i < *token_num; ++i) {
                if (consumed[i]) {
                    continue;
                }
                if (prev_active != -1) {
                    const int rank = get_pair_rank_from_ids(
                        merges_map, tokens[prev_active], tokens[i]);
                    if (rank != -1 && rank > best_rank) {
                        best_rank = rank;
                        best_left = prev_active;
                        best_right = i;
                    }
                }
                prev_active = i;
            }

            if (best_left == -1) {
                break;
            }

            struct MergeRule lookup_key = {.left_id = tokens[best_left],
                                            .right_id = tokens[best_right]};
            const struct MergeRule* rule = hashmap_get(merges_map, &lookup_key);
            if (!rule) {
                continue;
            }
            tokens[best_left] = rule->merge_id;
            consumed[best_right] = true;
        }

        int final_token_count = 0;
        for (int i = 0; i < *token_num; ++i) {
            if (!consumed[i]) {
                tokens[final_token_count++] = tokens[i];
            }
        }

        *token_num = final_token_count;
        free_buffer(consumed, use_arena);
    }
}

void encode(struct EncodeTask* task) {
    struct Arena arena;
    if (!arena_create(&arena, FIXED_ARENA_SIZE)) {
        log_debug("Error: Failed to create arena for encoding.");
        task->error_msg = "Memory allocation failed for arena.";
        return;
    }

    log_debug("Starting encode function with text: %s and pattern: %s",
              task->text, task->ctx->pattern);

    regex_t regex;
    struct ParserState parser;
    bool use_regex = task->ctx->pattern != NULL;
    if (use_regex) {
        if (regcomp(&regex, task->ctx->pattern, REG_EXTENDED) == true) {
            log_debug("Error: Regex could not be compiled.");
            task->error_msg = "Regex could not be compiled.";
            arena_destroy(&arena);
            return;
        }
    } else {
        parser = parser_init(task->text);
    }

    const char* cursor = task->text;
    bool add_prefix = cursor[0] != ' ';
    bool add_prefix_token = !add_prefix;

    while (true) {
        struct TokenSlice word_slice;
        bool has_token = false;

        if (use_regex) {
            regmatch_t match;
            if (regexec(&regex, cursor, 1, &match, 0) == 0) {
                word_slice.start = cursor + match.rm_so;
                word_slice.length = match.rm_eo - match.rm_so;
                has_token = true;
            }
        } else {
            if (parser_next_token(&parser, &word_slice)) {
                has_token = true;
            }
        }

        if (!has_token) {
            break;
        }

        // If the regex finds a zero-length match, word_len will be 0.
        // This would lead to calling `bpe_encode` with unitialized arrays, or
        // the `cursor` not advancing to the next step.
        if (word_slice.length == 0) {
            if (*(word_slice.start) == '\0') {
                break;
            }
            if (use_regex) {
                cursor = word_slice.start + 1;
            }
            continue;
        }

        size_t estimated_needed = word_slice.length * BPE_ARENA_MULTIPLIER;
        if (estimated_needed > arena.total_size) {
            task->error_msg =
                "A single word in the input text is too large to be processed.";
            break;
        }

        if (arena.current_offset + estimated_needed > arena.total_size) {
            log_debug("Resetting arena before processing word of length %zu",
                      word_slice.length);
            arena_reset(&arena);
        }

        char* word = arena_alloc(&arena, word_slice.length + 1);
        memcpy(word, word_slice.start, word_slice.length);
        word[word_slice.length] = '\0';
        log_debug("Matched word: length=%zu, word='%s'", word_slice.length,
                  word);

        if (add_prefix_token && task->ctx->prefix) {
            log_debug("Adding encoded prefix to tokens");
            bool prefix_in_arena = task->ctx->use_pretokenizer &&
                                   task->ctx->use_arena;
            char* prefix_encoded = prefix_in_arena
                ? pretokenizer_encode_arena(
                      &arena, task->ctx->prefix,
                      (const char**)task->ctx->special_chars, NULL,
                      task->ctx->is_byte_encoder)
                : pretokenizer_encode(
                      task->ctx->prefix,
                      (const char**)task->ctx->special_chars, NULL,
                      task->ctx->is_byte_encoder);

            if (!prefix_encoded) {
                task->error_msg = "Failed to encode prefix.";
                if (!prefix_in_arena) {
                    free(prefix_encoded);
                }
                break;
            }

            struct Boundary prefix_boundaries[strlen(prefix_encoded)];
            int prefix_tokens[strlen(prefix_encoded)];
            int pcount = 0;

            for (char* ptr = prefix_encoded; *ptr != '\0';
                 ptr += utf8_char_length((unsigned char*)ptr)) {
                int clen = utf8_char_length((unsigned char*)ptr);
                struct Boundary b = {.start = ptr, .end = ptr + clen - 1};
                prefix_boundaries[pcount++] = b;
            }

            bpe_encode_string(&arena, task->ctx->use_arena,
                              task->ctx->use_bpe_optimized,
                              task->ctx->vocab_encode, prefix_boundaries,
                              prefix_tokens, &pcount);

            vector_append_array(task->tokens, prefix_tokens, pcount);
            log_debug("Encoded %d prefix tokens.", pcount);

            if (!prefix_in_arena) {
                free(prefix_encoded);
            }
            add_prefix_token = false;
        }

        bool encoded_in_arena = task->ctx->use_pretokenizer &&
                                task->ctx->use_arena;
        char* encoded_word = encoded_in_arena
            ? pretokenizer_encode_arena(
                  &arena, word, (const char**)task->ctx->special_chars,
                  add_prefix ? task->ctx->prefix : NULL,
                  task->ctx->is_byte_encoder)
            : pretokenizer_encode(
                  word, (const char**)task->ctx->special_chars,
                  add_prefix ? task->ctx->prefix : NULL,
                  task->ctx->is_byte_encoder);
        add_prefix = false;

        if (!encoded_word) {
            task->error_msg = "Failed to encode word.";
            break;
        }

        size_t encoded_len = strlen(encoded_word);
        int word_tokens[encoded_len > 0 ? encoded_len : 1];
        int word_tokens_num = 0;

        if (task->ctx->merges_map != NULL) {
            log_debug("Using ID-based BPE encoding path.");

            for (char* ptr = encoded_word; *ptr != '\0';) {
                int char_len = utf8_char_length((unsigned char*)ptr);
                char temp_char[char_len + 1];
                memcpy(temp_char, ptr, char_len);
                temp_char[char_len] = '\0';

                const struct Token* found = hashmap_get(
                    task->ctx->vocab_encode, &(struct Token){.key = temp_char});
                if (found) {
                    word_tokens[word_tokens_num++] = found->value;
                } else {
                    word_tokens[word_tokens_num++] = -1;
                }
                ptr += char_len;
            }

            bpe_encode_ids(&arena, task->ctx->use_arena,
                           task->ctx->use_bpe_optimized,
                           task->ctx->merges_map, word_tokens,
                           &word_tokens_num);
        } else {
            log_debug("Using string-based BPE encoding path.");
            struct Boundary
                word_token_boundaries[encoded_len > 0 ? encoded_len : 1];

            for (char* ptr = encoded_word; *ptr != '\0';) {
                int token_len = next_token_length(ptr);
                word_token_boundaries[word_tokens_num++] =
                    (struct Boundary){.start = ptr, .end = ptr + token_len - 1};
                ptr += token_len;
            }

            bpe_encode_string(&arena, task->ctx->use_arena,
                              task->ctx->use_bpe_optimized,
                              task->ctx->vocab_encode,
                              word_token_boundaries, word_tokens,
                              &word_tokens_num);
        }

        if (!encoded_in_arena) {
            free(encoded_word);
        }

        vector_append_array(task->tokens, word_tokens, word_tokens_num);
        log_debug("Appended %d word tokens.", word_tokens_num);

        if (use_regex) {
            cursor = word_slice.start + word_slice.length;
        }
    }

    task->error_msg = NULL;

    if (use_regex) {
        regfree(&regex);
    }
    log_debug("Completed encode function. Total tokens: %lu",
              task->tokens->size);
    arena_destroy(&arena);
}

void decode(struct DecodeTask* task) {
    log_debug("Entered decode function");

    int token_num = *task->tokens_size;
    log_debug("Number of tokens to decode: %d", token_num);

    size_t total_size = 0;
    for (int i = 0; i < token_num; ++i) {
        int token_id = task->tokens[i];
        if (token_id < 0 || token_id >= task->ctx->vocab_size_decode) {
            log_debug("Token value %d is out of bounds (vocab size = %d).",
                      token_id, task->ctx->vocab_size_decode);
            int msg_len = snprintf(
                NULL,
                0,
                "Invalid token at index %d: %d. Element must be non-negative and less than vocab size.",
                i,
                token_id);
            char* msg = malloc((size_t)msg_len + 1);
            if (msg) {
                snprintf(
                    msg,
                    (size_t)msg_len + 1,
                    "Invalid token at index %d: %d. Element must be non-negative and less than vocab size.",
                    i,
                    token_id);
                task->error_msg = msg;
                task->error_msg_owned = true;
            } else {
                task->error_msg = 
                    "Invalid token. Element must be non-negative and less than vocab size.";
                task->error_msg_owned = false;
            }
            task->result = NULL;
            return;
        }
        total_size += task->ctx->vocab_decode_lens[token_id];
    }
    log_debug("Calculated total size for decoded string: %zu", total_size);

    char* text = (char*)malloc(total_size + 1);

    if (!text) {
        log_debug("Error: Memory allocation failed for text buffer");
        task->error_msg = "Failed to allocate memory for text buffer";
        task->result = NULL;
        return;
    }

    text[0] = '\0';
    log_debug("Allocated final buffer of size %zu bytes", total_size + 1);

    char* write_ptr = text;

    for (int i = 0; i < token_num; i++) {
        log_debug("Processing token at index %d", i);

        int item = task->tokens[i];
        const char* word = task->ctx->vocab_decode[item];

        size_t word_len = strlen(word);
        memcpy(write_ptr, word, word_len);
        write_ptr += word_len;

        log_debug("Copied word '%s' to buffer.", word);
    }

    *write_ptr = '\0';
    log_debug("Final raw decoded string: '%s'", text);

    char* decoded_text = (char*)malloc(total_size + 1);
    if (!decoded_text) {
        log_debug(
            "Error: Memory allocation failed for final decoded_text buffer");
        task->error_msg = "Failed to allocate memory for final text buffer";
        task->result = NULL;
        free(text);
        return;
    }

    size_t final_len = pretokenizer_decode(text, task->ctx, decoded_text);
    log_debug("Final decoded text: %s, Length: %zu", decoded_text, final_len);

    free(text);

    task->result = decoded_text;
    task->error_msg = NULL;
}

#ifdef USE_FOMA

PyObject* initialize_foma(void) {
    log_debug("Starting foma initialization");

    struct fsm* net = fsm_read_binary_file("./bin/hu.foma.bin");

    if (!net) {
        log_debug("Error: Failed to read the finite state machine");
        PyErr_SetString(PyExc_FileNotFoundError,
                        "Failed to read the finite state machine");
        return NULL;
    }

    struct apply_handle* handle = apply_init(net);

    if (!handle) {
        log_debug("Error: Couldn't initialize apply_handle");
        PyErr_SetString(PyExc_ValueError, "Couldn't initialize apply_handle.");
        return NULL;
    }

    return PyCapsule_New(handle, "foma.apply_handle", NULL);
}

PyObject* look_up_word(struct apply_handle* handle,
                       char* word,
                       bool only_longest) {
    log_debug("looking up word: %s", word);
    log_debug("Only longest morpheme splitting required");

    PyObject* py_list = PyList_New(0);
    char* split_morphemes = NULL;
    int max_morpheme_count = 0;

    while ((split_morphemes = apply_up(handle, word)) != NULL) {
        log_debug("found result: %s", split_morphemes);

        if (only_longest) {
            int morpheme_count = count_char(split_morphemes, '[');
            if (morpheme_count > max_morpheme_count) {
                max_morpheme_count = morpheme_count;
            } else {
                word = NULL;
                continue;
            }
        }

        PyObject* morpheme_list = PyList_New(0);
        size_t tmp_len = strlen(split_morphemes) + 1;
        char* tmp = (char*)malloc(tmp_len);

        if (!tmp) {
            log_debug("Error: Memory allocation failed for morpheme splitting");
            PyErr_SetString(PyExc_MemoryError,
                            "Couldn't allocate memory for morpheme splitting.");
            return NULL;
        }

        strcpy(tmp, split_morphemes);

        char* token = strtok(tmp, "[]");
        int should_add = 1;
        while (token != NULL) {
            if (should_add % 2 && strlen(token) > 0) {
                if (PyList_Append(morpheme_list, PyUnicode_FromString(token)) <
                    0) {
                    log_debug("Error: Failed to append token to morpheme_list");
                    PyErr_SetString(PyExc_RuntimeError,
                                    "Failed to append token to morpheme_list.");
                    free(tmp);
                    return NULL;
                }
            }
            should_add++;
            token = strtok(NULL, "[]");
        }
        free(tmp);

        if (only_longest) {
            if (PyList_Size(py_list) == 0) {
                if (PyList_Append(py_list, morpheme_list) < 0) {
                    log_debug(
                        "Error: Failed to append morpheme_list to py_list");
                    PyErr_SetString(
                        PyExc_RuntimeError,
                        "Failed to append morpheme_list to py_list.");
                    Py_DECREF(morpheme_list);
                    return NULL;
                }
            } else {
                if (PyList_SetItem(py_list, 0, morpheme_list) < 0) {
                    log_debug("Error: Failed to set py_list item");
                    PyErr_SetString(PyExc_RuntimeError,
                                    "Failed to set py_list item.");
                    Py_DECREF(morpheme_list);
                    return NULL;
                }
            }
        } else {
            if (PyList_Append(py_list, morpheme_list) < 0) {
                log_debug("Error: Failed to append morpheme_list to py_list");
                PyErr_SetString(PyExc_RuntimeError,
                                "Failed to append morpheme_list to py_list.");
                Py_DECREF(morpheme_list);
                return NULL;
            }
        }

        word = NULL;
    }

    return py_list;
}

#endif

static int get_pair_rank_from_strings(const struct HashMap* vocab,
                                      const struct Boundary token_boundaries[],
                                      const int left_idx,
                                      const int right_idx) {
    const ptrdiff_t left_len =
        (token_boundaries[left_idx].end - token_boundaries[left_idx].start) + 1;
    const ptrdiff_t right_len =
        (token_boundaries[right_idx].end - token_boundaries[right_idx].start) +
        1;

    const ptrdiff_t pair_len = left_len + right_len;
    char pair_str[pair_len + 1];
    memcpy(pair_str, token_boundaries[left_idx].start, left_len);
    memcpy(pair_str + left_len, token_boundaries[right_idx].start, right_len);
    pair_str[pair_len] = '\0';

    const struct Token* found_token =
        hashmap_get((struct HashMap*)vocab, &(struct Token){.key = pair_str});

    log_debug("pair_str='%s'", pair_str);

    return (found_token != NULL) ? found_token->value : -1;
}

static int get_pair_rank_from_ids(const struct HashMap* merges_map,
                                  const int left_id,
                                  const int right_id) {
    struct MergeRule key = {.left_id = left_id, .right_id = right_id};
    const struct MergeRule* found_item =
        hashmap_get((struct HashMap*)merges_map, &key);

    if (found_item == NULL) {
        return -1;
    }

    return (found_item != NULL) ? found_item->rank : -1;
}
