#include "hutoken/core.h"

#include "Python.h"
#ifdef USE_FOMA
#include "fomalib.h"
#endif
#include "listobject.h"
#include "unicodeobject.h"

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

void bpe_encode_arena_string(struct Arena* arena,
                             struct HashMap* vocab,
                             struct Boundary token_boundaries[],
                             int tokens[],
                             int* token_num) {
    struct MinPQ pq;
    if (min_pq_init_arena(arena, &pq, *token_num) != MIN_PQ_SUCCESS) {
        log_debug("Failed to initialize priority queue.");
        return;
    }

    // Before using the min-priority queue, we use a linked list to track the
    // sequence of active tokens. This is because invalidating the tokens after
    // a merge is inefficient, while tracking the active tokens with a linked
    // list is not.
    struct TokenNode* nodes =
        arena_alloc(arena, *token_num * sizeof(struct TokenNode));
    bool* consumed = arena_alloc(arena, *token_num * sizeof(bool));
    memset(consumed, 0, *token_num * sizeof(bool));
    if (!nodes || !consumed) {
        log_debug("Failed to allocate memory for token nodes.");
        return;
    }

    for (int i = 0; i < *token_num; ++i) {
        nodes[i].prev = i - 1;
        nodes[i].next = i + 1;
    }
    nodes[*token_num - 1].next = -1;

    for (int i = 0; i < *token_num - 1; ++i) {
        const int rank =
            get_pair_rank_from_strings(vocab, token_boundaries, i, i + 1);
        if (rank != -1) {
            const struct MergeCandidate candidate = {
                .rank = rank, .left_idx = i, .right_idx = i + 1};

            if (min_pq_push_arena(arena, &pq, candidate) != MIN_PQ_SUCCESS) {
                log_debug("Failed to push to queue.");
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
            // The pair is stale.
            continue;
        }

        const int current_rank = get_pair_rank_from_strings(
            vocab, token_boundaries, left_idx, right_idx);

        if (best_pair.rank != current_rank) {
            // It is possible that the right token has been modified, and the
            // queue does not recognize that, which passes every other check.
            // This way, if it has been modified, the rank is different as well,
            // and the merge candidate is skipped.
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
                vocab, token_boundaries, prev_idx,
                left_idx);  // NOLINT: readability-suspicious-call-argument

            if (rank != -1) {
                min_pq_push_arena(
                    arena, &pq,
                    (struct MergeCandidate){.rank = rank,
                                            .left_idx = prev_idx,
                                            .right_idx = left_idx});
            }
        }

        if (next_idx != -1) {
            const int rank = get_pair_rank_from_strings(vocab, token_boundaries,
                                                        left_idx, next_idx);

            if (rank != -1) {
                min_pq_push_arena(
                    arena, &pq,
                    (struct MergeCandidate){.rank = rank,
                                            .left_idx = left_idx,
                                            .right_idx = next_idx});
            }
        }
    }

    struct Boundary* final_boundaries =
        arena_alloc(arena, *token_num * sizeof(struct Boundary));
    if (!final_boundaries) {
        log_debug("Failed to allocate memory for final boundaries.");
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
        log_debug("rank=%d", tokens[i]);
    }
}

void bpe_encode_arena_ids(struct Arena* arena,
                          struct HashMap* merges_map,
                          int tokens[],
                          int* token_num) {
    struct MinPQ pq;
    if (min_pq_init_arena(arena, &pq, *token_num) != MIN_PQ_SUCCESS) {
        log_debug("Failed to initialize priority queue.");
        return;
    }

    // Before using the min-priority queue, we use a linked list to track the
    // sequence of active tokens. This is because invalidating the tokens after
    // a merge is inefficient, while tracking the active tokens with a linked
    // list is not.
    struct TokenNode* nodes =
        arena_alloc(arena, *token_num * sizeof(struct TokenNode));
    bool* consumed = arena_alloc(arena, *token_num * sizeof(bool));
    memset(consumed, 0, *token_num * sizeof(bool));
    if (!nodes || !consumed) {
        log_debug("Failed to allocate memory for token nodes.");
        return;
    }

    for (int i = 0; i < *token_num; ++i) {
        nodes[i].prev = i - 1;
        nodes[i].next = i + 1;
    }
    nodes[*token_num - 1].next = -1;

    for (int i = 0; i < *token_num - 1; ++i) {
        const int rank =
            get_pair_rank_from_ids(merges_map, tokens[i], tokens[i + 1]);
        if (rank != -1) {
            const struct MergeCandidate candidate = {
                .rank = rank, .left_idx = i, .right_idx = i + 1};

            if (min_pq_push_arena(arena, &pq, candidate) != MIN_PQ_SUCCESS) {
                log_debug("Failed to push to queue.");
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
            // The pair is stale.
            continue;
        }

        const int current_rank = get_pair_rank_from_ids(
            merges_map, tokens[left_idx], tokens[right_idx]);

        if (best_pair.rank != current_rank) {
            // It is possible that the right token has been modified, and the
            // queue does not recognize that, which passes every other check.
            // This way, if it has been modified, the rank is different as well,
            // and the merge candidate is skipped.
            continue;
        }

        struct MergeRule key = {.left_id = tokens[left_idx],
                                .right_id = tokens[right_idx]};
        const struct MergeRule* rule = hashmap_get(merges_map, &key);
        if (rule) {
            tokens[left_idx] = rule->merge_id;
            consumed[right_idx] = true;
        } else {
            continue;
        }

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
                min_pq_push_arena(
                    arena, &pq,
                    (struct MergeCandidate){.rank = rank,
                                            .left_idx = prev_idx,
                                            .right_idx = left_idx});
            }
        }

        if (next_idx != -1) {
            const int rank = get_pair_rank_from_ids(
                merges_map, tokens[left_idx], tokens[next_idx]);

            if (rank != -1) {
                min_pq_push_arena(
                    arena, &pq,
                    (struct MergeCandidate){.rank = rank,
                                            .left_idx = left_idx,
                                            .right_idx = next_idx});
            }
        }
    }

    int final_token_count = 0;
    for (int i = 0; i < *token_num; ++i) {
        if (!consumed[i]) {
            tokens[final_token_count++] = tokens[i];
        }
    }

    *token_num = final_token_count;
}

void encode(struct EncodeTask* task) {
    struct Arena arena;
    const size_t text_len = strlen(task->text);
    const size_t arena_size = text_len * 64 > 8192 ? text_len * 64 : 8192;
    if (!arena_create(&arena, arena_size)) {
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

        char* word = arena_alloc(&arena, word_slice.length + 1);
        memcpy(word, word_slice.start, word_slice.length);
        word[word_slice.length] = '\0';
        log_debug("Matched word: length=%zu, word='%s'", word_slice.length,
                  word);

        if (add_prefix_token && task->ctx->prefix) {
            log_debug("Adding encoded prefix to tokens");
            char* prefix_encoded = pretokenizer_encode_arena(
                &arena, task->ctx->prefix, (const char**)task->ctx->special_chars, NULL,
                task->ctx->is_byte_encoder);

            struct Boundary prefix_boundaries[strlen(prefix_encoded)];
            int prefix_tokens[strlen(prefix_encoded)];
            int pcount = 0;

            for (char* ptr = prefix_encoded; *ptr != '\0';
                 ptr += utf8_char_length((unsigned char*)ptr)) {
                int clen = utf8_char_length((unsigned char*)ptr);
                struct Boundary b = {.start = ptr, .end = ptr + clen - 1};
                prefix_boundaries[pcount++] = b;
            }

            bpe_encode_arena_string(&arena, task->ctx->vocab_encode, prefix_boundaries,
                       prefix_tokens, &pcount);

            for (int i = 0; i < pcount; i++) {
                task->tokens[*task->tokens_size + i] = prefix_tokens[i];
                log_debug("Encoded prefix token: %d", prefix_tokens[i]);
            }
            *task->tokens_size += pcount;

            add_prefix_token = false;
        }

        char* encoded_word = pretokenizer_encode_arena(
            &arena, word, (const char**)task->ctx->special_chars,
            add_prefix ? task->ctx->prefix : NULL, task->ctx->is_byte_encoder);
        add_prefix = false;

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

            bpe_encode_arena_ids(&arena, task->ctx->merges_map, word_tokens,
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

            bpe_encode_arena_string(&arena, task->ctx->vocab_encode,
                                    word_token_boundaries, word_tokens,
                                    &word_tokens_num);
        }

        for (int i = 0; i < word_tokens_num; i++) {
            task->tokens[i + *task->tokens_size] = word_tokens[i];
            log_debug("Encoded token: %d", word_tokens[i]);
        }

        *task->tokens_size += word_tokens_num;

        if (use_regex) {
            cursor = word_slice.start + word_slice.length;
        }
    }

    task->error_msg = NULL;

    if (use_regex) {
        regfree(&regex);
    }
    log_debug("Completed encode function. Total tokens: %d",
              *task->tokens_size);
    arena_destroy(&arena);
}

void decode(struct DecodeTask* task) {
    log_debug("Entered decode function");

    int token_num = *task->tokens_size;
    log_debug("Number of tokens to decode: %d", token_num);

    size_t text_size = sizeof(char) * (token_num + 1);
    char* text = (char*)malloc(text_size);

    if (!text) {
        log_debug("Error: Memory allocation failed for text buffer");
        task->error_msg = "Failed to allocate memory for text buffer";
        task->result = NULL;
        return;
    }

    text[0] = '\0';
    log_debug("Initialized text buffer to an empty string (size: %zu bytes)",
              text_size);

    for (int i = 0; i < token_num; i++) {
        log_debug("Processing token at index %d", i);

        int* token = &task->tokens[i];
        if (!token) {
            log_debug("Error: Token at index %d is not an integer", i);
            task->error_msg = "All elements of the list must be integers";
            free(text);
            task->result = NULL;
            return;
        }

        int item = *token;
        if (item < 0 || item >= task->ctx->vocab_size_decode) {
            log_debug(
                "Error: Token value %d is out of bounds (vocab_size = %d)",
                item, task->ctx->vocab_size_decode);
            task->error_msg =
                "Element must be non-negative and less than vocab size.";
            free(text);
            task->result = NULL;
            return;
        }

        const char* word = task->ctx->vocab_decode[item];
        size_t word_len = strlen(word);
        log_debug("Decoded token value %d to word '%s' (length: %zu)", item,
                  word, word_len);

        size_t current_len = strlen(text);
        if (current_len + word_len + 1 >= text_size) {
            log_debug(
                "Resizing text buffer: current length = %zu, word length = "
                "%zu, current buffer size = %zu",
                current_len, word_len, text_size);

            size_t buffer_size =
                current_len + TEXT_SIZE_INCREMENT + word_len + 1;
            char* new_text = realloc(text, buffer_size);

            if (new_text == NULL) {
                log_debug("Error: Memory reallocation failed for text buffer");
                free(text);
                task->error_msg = "Failed to allocate memory for text buffer";
                task->result = NULL;
                return;
            }
            text = new_text;
            text_size = buffer_size;

            log_debug("Resized text buffer to new size: %zu bytes",
                      buffer_size);
        }

        (void)strcat(text, word);
        log_debug(
            "Appended word '%s' to text buffer. Current text: '%s' (buffer "
            "size: %zu bytes)",
            word, text, text_size);
    }

    char* decoded_text =
        pretokenizer_decode(text, (const char**)task->ctx->special_chars,
                            task->ctx->prefix, task->ctx->is_byte_encoder);
    log_debug("Decoded_text: %s", decoded_text);

    /*
    log_debug(
        "Successfully created Python string from decoded text (UTF-8 encoded, "
        "might be wrong here): '%s'",
        decoded_text);*/

    task->result = strdup(decoded_text);
    task->error_msg = NULL;

    free(text);
    free(decoded_text);
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

    log_debug("found_item->merge_id=%d", found_item->merge_id);

    return (found_item != NULL) ? found_item->rank : -1;
}
