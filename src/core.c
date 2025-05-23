#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <regex.h>
#include <time.h>
#include <stdarg.h>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#include "helper.c"
#include "hashmap.c"

void bpe_encode(struct HashMap *vocab, Boundary token_boundaries[], int tokens[], int *token_num)
{
    if (DEBUG_ENABLED()) {
        log_debug("bpe_encode: Starting with %d tokens", *token_num);
    }

    if (unlikely(*token_num <= 1)) {
        // Fast path for single-token words
        if (*token_num == 1) {
            const uint8_t *start = token_boundaries[0].start;
            const uint8_t *end = token_boundaries[0].end;
            int len = (int)(end - start) + 1;
            
            char single_token[len + 1];
            memcpy(single_token, start, len);
            single_token[len] = '\0';
            
            if (DEBUG_ENABLED()) {
                log_debug("bpe_encode: Single token fast path, token='%s'", single_token);
            }
            
            struct Token probe = { .key = single_token, .value = 0 };
            tokens[0] = hashmap_get(vocab, &probe);
            
            if (DEBUG_ENABLED()) {
                log_debug("bpe_encode: Single token mapped to ID=%d", tokens[0]);
            }
        }
        return;
    }
    
    char pair_buffer[4096];
    bool merged;
    
    #define MAX_PAIRS 1024
    int pair_ranks[MAX_PAIRS];
    int original_token_num = *token_num;
    
    if (DEBUG_ENABLED()) {
        log_debug("bpe_encode: Using %s approach for %d tokens", 
                 original_token_num < MAX_PAIRS ? "pre-computed ranks" : "on-demand ranks", 
                 original_token_num);
    }
    
    if (original_token_num < MAX_PAIRS) {
        for (int i = 0; i < original_token_num - 1; i++) {
            pair_ranks[i] = INT_MAX;
        }
        
        // Pre-compute all pair ranks in one pass
        if (DEBUG_ENABLED()) {
            log_debug("bpe_encode: Pre-computing pair ranks for %d pairs", original_token_num - 1);
        }
        
        for (int i = 0; i < original_token_num - 1; i++) {
            const uint8_t *s1 = token_boundaries[i].start;
            const uint8_t *e1 = token_boundaries[i].end;
            int l1 = (int)(e1 - s1) + 1;

            const uint8_t *s2 = token_boundaries[i + 1].start;
            const uint8_t *e2 = token_boundaries[i + 1].end;
            int l2 = (int)(e2 - s2) + 1;

            int len = l1 + l2;
            if (likely(len < 4096)) {
                memcpy(pair_buffer, s1, l1);
                memcpy(pair_buffer + l1, s2, l2);
                pair_buffer[len] = '\0';

                struct Token probe = { .key = pair_buffer, .value = 0 };
                int rank = hashmap_get(vocab, &probe);
                pair_ranks[i] = (rank != -1) ? rank : INT_MAX;
                
                if (DEBUG_ENABLED() && rank != -1 && i < 5) {
                    // Only log first few pairs to avoid flooding log
                    log_debug("bpe_encode: Pair[%d]='%s', rank=%d", i, pair_buffer, rank);
                }
            }
        }
    }
    
    int iteration = 0;
    int total_merges = 0;
    
    do {
        iteration++;
        merged = false;
        int min_idx = -1;
        int min_rank = INT_MAX;

        if (*token_num < MAX_PAIRS) {
            // Fast path: Use pre-computed ranks
            for (int i = 0; i < *token_num - 1; i++) {
                if (pair_ranks[i] < min_rank) {
                    min_idx = i;
                    min_rank = pair_ranks[i];
                }
            }
        } else {
            // Fallback for very long tokens: compute on demand
            if (DEBUG_ENABLED()) {
                log_debug("bpe_encode: Computing ranks on-demand for %d tokens", *token_num);
            }
            
            for (int i = 0; i < *token_num - 1; i++) {
                const uint8_t *s1 = token_boundaries[i].start;
                const uint8_t *e1 = token_boundaries[i].end;
                int l1 = (int)(e1 - s1) + 1;

                const uint8_t *s2 = token_boundaries[i + 1].start;
                const uint8_t *e2 = token_boundaries[i + 1].end;
                int l2 = (int)(e2 - s2) + 1;

                int len = l1 + l2;
                if (likely(len < 4096)) {
                    memcpy(pair_buffer, s1, l1);
                    memcpy(pair_buffer + l1, s2, l2);
                    pair_buffer[len] = '\0';

                    struct Token probe = { .key = pair_buffer, .value = 0 };
                    int rank = hashmap_get(vocab, &probe);
                    if (rank != -1 && rank < min_rank) {
                        min_idx = i;
                        min_rank = rank;
                    }
                }
            }
        }

        if (min_rank != INT_MAX) {
            merged = true;
            total_merges++;
            
            if (DEBUG_ENABLED() && (total_merges <= 5 || total_merges % 100 == 0)) {
                const uint8_t *s1 = token_boundaries[min_idx].start;
                const uint8_t *e1 = token_boundaries[min_idx].end;
                int l1 = (int)(e1 - s1) + 1;
                const uint8_t *s2 = token_boundaries[min_idx + 1].start;
                const uint8_t *e2 = token_boundaries[min_idx + 1].end;
                int l2 = (int)(e2 - s2) + 1;
                
                char debug_buffer1[l1 + 1];
                char debug_buffer2[l2 + 1];
                memcpy(debug_buffer1, s1, l1);
                memcpy(debug_buffer2, s2, l2);
                debug_buffer1[l1] = '\0';
                debug_buffer2[l2] = '\0';
                
                log_debug("bpe_encode: Merge #%d: '%s' + '%s' at position %d, rank=%d", 
                         total_merges, debug_buffer1, debug_buffer2, min_idx, min_rank);
            }
            
            token_boundaries[min_idx].end = token_boundaries[min_idx + 1].end;
            
            // Fast memmove for token boundaries
            if (likely(*token_num - min_idx - 2 > 0)) {
                memmove(&token_boundaries[min_idx + 1], 
                        &token_boundaries[min_idx + 2],
                        (*token_num - min_idx - 2) * sizeof(Boundary));
            }
            
            // Update pair_ranks if using the fast path
            if (*token_num < MAX_PAIRS) {
                if (min_idx > 0) {
                    const uint8_t *s1 = token_boundaries[min_idx-1].start;
                    const uint8_t *e1 = token_boundaries[min_idx-1].end;
                    int l1 = (int)(e1 - s1) + 1;

                    const uint8_t *s2 = token_boundaries[min_idx].start;
                    const uint8_t *e2 = token_boundaries[min_idx].end;
                    int l2 = (int)(e2 - s2) + 1;

                    int len = l1 + l2;
                    if (likely(len < 4096)) {
                        memcpy(pair_buffer, s1, l1);
                        memcpy(pair_buffer + l1, s2, l2);
                        pair_buffer[len] = '\0';

                        struct Token probe = { .key = pair_buffer, .value = 0 };
                        int rank = hashmap_get(vocab, &probe);
                        pair_ranks[min_idx-1] = (rank != -1) ? rank : INT_MAX;
                        
                        if (DEBUG_ENABLED() && rank != -1 && total_merges <= 3) {
                            log_debug("bpe_encode: Updated left context pair rank to %d", rank);
                        }
                    } else {
                        pair_ranks[min_idx-1] = INT_MAX;
                    }
                }
                
                for (int i = min_idx; i < *token_num - 2; i++) {
                    pair_ranks[i] = pair_ranks[i+1];
                }
                
                // Compute new rank for the last available position
                if (min_idx < *token_num - 2) {
                    const uint8_t *s1 = token_boundaries[min_idx].start;
                    const uint8_t *e1 = token_boundaries[min_idx].end;
                    int l1 = (int)(e1 - s1) + 1;

                    const uint8_t *s2 = token_boundaries[min_idx+1].start;
                    const uint8_t *e2 = token_boundaries[min_idx+1].end;
                    int l2 = (int)(e2 - s2) + 1;

                    int len = l1 + l2;
                    if (likely(len < 4096)) {
                        memcpy(pair_buffer, s1, l1);
                        memcpy(pair_buffer + l1, s2, l2);
                        pair_buffer[len] = '\0';

                        struct Token probe = { .key = pair_buffer, .value = 0 };
                        int rank = hashmap_get(vocab, &probe);
                        pair_ranks[min_idx] = (rank != -1) ? rank : INT_MAX;
                        
                        if (DEBUG_ENABLED() && rank != -1 && total_merges <= 3) {
                            log_debug("bpe_encode: Updated right context pair rank to %d", rank);
                        }
                    } else {
                        pair_ranks[min_idx] = INT_MAX;
                    }
                }
            }
            
            (*token_num)--;
        }
    } while (merged);

    if (DEBUG_ENABLED()) {
        log_debug("bpe_encode: Completed %d merge iterations, performed %d total merges, final token count=%d",
                 iteration, total_merges, *token_num);
    }

    // Process tokens in a single pass with cached lookups
    if (DEBUG_ENABLED()) {
        log_debug("bpe_encode: Converting %d boundaries to token IDs", *token_num);
    }
    
    for (int i = 0; i < *token_num; i++) {
        const uint8_t *start = token_boundaries[i].start;
        const uint8_t *end = token_boundaries[i].end;
        int len = (int)(end - start) + 1;

        if (unlikely(len >= 4096)) {
            tokens[i] = -1;
            if (DEBUG_ENABLED()) {
                log_debug("bpe_encode: Warning - token at index %d exceeds buffer size (len=%d)", i, len);
            }
            continue;
        }

        memcpy(pair_buffer, start, len);
        pair_buffer[len] = '\0';

        struct Token probe = { .key = pair_buffer, .value = 0 };
        tokens[i] = hashmap_get(vocab, &probe);
    }
}

void encode(const uint8_t *text, struct HashMap *vocab, regex_t *regex, int tokens[], int *tokens_size) {
    #define MAX_CACHE_STR_LEN 128
    size_t text_len = strlen((const char*)text);
    
    if (DEBUG_ENABLED()) {
        if (text_len < 50) {
            log_debug("encode: Starting with text '%s' (length %zu)", text, text_len);
        } else {
            // Only show beginning of text for long strings
            char preview[51] = {0};
            memcpy(preview, text, 50);
            log_debug("encode: Starting with text '%.50s...' (length %zu)", preview, text_len);
        }
    }
    
    // Try cache for small strings
    if (likely(text_len <= MAX_CACHE_STR_LEN)) {
        int token_count = 0;
        int* cached_tokens = lookup_cache((const char*)text, &token_count);
        if (cached_tokens) {
            memcpy(tokens, cached_tokens, token_count * sizeof(int));
            *tokens_size = token_count;
            
            if (DEBUG_ENABLED()) {
                log_debug("encode: Cache hit! Returning %d tokens", token_count);
            }
            return;
        }
    }

    if (DEBUG_ENABLED()) {
        log_debug("encode: Processing text of length %zu", text_len);
    }
    
    regmatch_t match;
    const uint8_t *cursor = text;
    const uint8_t *end = text + text_len;

    // Stack allocate buffers for common case
    #define MAX_WORD_LEN 1024
    Boundary stack_boundaries[MAX_WORD_LEN];
    int stack_tokens[MAX_WORD_LEN];

    *tokens_size = 0;
    int max_tokens = 65536;
    
    // Main tokenization loop - optimized for most common path
    while (likely(cursor < end && *tokens_size < max_tokens)) {
        if (unlikely(regexec(regex, (const char *)cursor, 1, &match, 0) != 0)) {
            if (DEBUG_ENABLED()) {
                log_debug("encode: Regex match failed, breaking at position %ld", cursor - text);
            }
            break;
        }
            
        int word_start = match.rm_so;
        int word_end = match.rm_eo;
        int word_len = word_end - word_start;

        if (unlikely(word_len <= 0)) {
            if (DEBUG_ENABLED()) {
                log_debug("encode: Skipping zero or negative length match (%d) at position %ld", 
                          word_len, cursor - text);
            }
            cursor += (word_end > 0) ? word_end : 1;
            continue;
        }

        // Fast stack allocation path
        Boundary *word_token_boundaries = stack_boundaries;
        int *word_tokens = stack_tokens;
        
        // Only allocate heap for unusually long words
        if (unlikely(word_len > MAX_WORD_LEN)) {
            if (DEBUG_ENABLED()) {
                log_debug("encode: Using heap allocation for large word (len=%d)", word_len);
            }
            word_token_boundaries = malloc(word_len * sizeof(Boundary));
            word_tokens = malloc(word_len * sizeof(int));
            if (unlikely(!word_token_boundaries || !word_tokens)) {
                if (DEBUG_ENABLED()) {
                    log_debug("encode: Memory allocation failed for word_len=%d", word_len);
                }
                if (word_token_boundaries) free(word_token_boundaries);
                if (word_tokens) free(word_tokens);
                return;
            }
        }

        // Performance optimization: unroll small loops, use direct assignment
        if (likely(word_len <= 16)) {
            #define INIT_BOUNDARY(i) \
                word_token_boundaries[i].start = cursor + word_start + i; \
                word_token_boundaries[i].end = cursor + word_start + i;
                
            // Manual loop unrolling for extremely common small token case
            if (likely(word_len <= 8)) {
                if (word_len > 0) INIT_BOUNDARY(0);
                if (word_len > 1) INIT_BOUNDARY(1);
                if (word_len > 2) INIT_BOUNDARY(2);
                if (word_len > 3) INIT_BOUNDARY(3);
                if (word_len > 4) INIT_BOUNDARY(4);
                if (word_len > 5) INIT_BOUNDARY(5);
                if (word_len > 6) INIT_BOUNDARY(6);
                if (word_len > 7) INIT_BOUNDARY(7);
            } else {
                for (int i = 0; i < word_len; i++) {
                    INIT_BOUNDARY(i);
                }
            }
            #undef INIT_BOUNDARY
        } else {
            // For larger arrays, use normal loop 
            for (int i = 0; i < word_len; i++) {
                word_token_boundaries[i].start = cursor + word_start + i;
                word_token_boundaries[i].end = cursor + word_start + i;
            }
        }

        int word_token_num = word_len;
        
        if (DEBUG_ENABLED()) {
            log_debug("encode: Calling bpe_encode with %d characters", word_len);
        }
        
        bpe_encode(vocab, word_token_boundaries, word_tokens, &word_token_num);
        
        if (DEBUG_ENABLED()) {
            log_debug("encode: bpe_encode returned %d tokens", word_token_num);
        }

        int tokens_to_copy = word_token_num;
        if (unlikely(*tokens_size + tokens_to_copy > max_tokens)) {
            tokens_to_copy = max_tokens - *tokens_size;
            
            if (DEBUG_ENABLED()) {
                log_debug("encode: Token limit reached! Truncating from %d to %d tokens", 
                         word_token_num, tokens_to_copy);
            }
        }

        memcpy(tokens + *tokens_size, word_tokens, tokens_to_copy * sizeof(int));
        *tokens_size += tokens_to_copy;

        if (unlikely(word_token_boundaries != stack_boundaries)) {
            free(word_token_boundaries);
            free(word_tokens);
        }

        cursor += word_end;
    }

    if (likely(text_len <= MAX_CACHE_STR_LEN)) {
        update_cache((const char*)text, tokens, *tokens_size);
    }
}

PyObject *decode(PyObject *tokens, char **vocab_decode, int vocab_size)
{
    Py_ssize_t token_num = PyList_Size(tokens);
    
    if (DEBUG_ENABLED()) {
        log_debug("decode: Starting with %zd tokens", token_num);
    }
    
    // Fast path for empty token list
    if (unlikely(token_num == 0)) {
        if (DEBUG_ENABLED()) {
            log_debug("decode: Empty token list, returning empty string");
        }
        return PyUnicode_FromString("");
    }
    
    int *token_ids = malloc(token_num * sizeof(int));
    if (unlikely(!token_ids)) {
        if (DEBUG_ENABLED()) {
            log_debug("decode: Memory allocation failed for token_ids array");
        }
        return PyErr_NoMemory();
    }
    
    size_t total_len = 0;
    bool need_validation = true;
    
    // First pass: validate and calculate size in one loop
    if (DEBUG_ENABLED()) {
        log_debug("decode: Pass 1 - Validating tokens and calculating total length");
    }
    
    for (Py_ssize_t i = 0; i < token_num; i++) {
        PyObject *token = PyList_GetItem(tokens, i);
        
        // Fast-path for token integer check
        if (likely(PyLong_Check(token))) {
            int item = (int)PyLong_AsLong(token);
            
            if (likely(item >= 0 && item < vocab_size)) {
                token_ids[i] = item;
                total_len += strlen(vocab_decode[item]);
                continue;
            }
            
            if (DEBUG_ENABLED()) {
                log_debug("decode: Invalid token ID %d at position %zd", item, i);
            }
            
            free(token_ids);
            PyErr_SetString(PyExc_ValueError, 
                "Token ID out of range (must be 0 <= id < vocab_size)");
            return NULL;
        }
        
        // Slow path - handle error
        if (DEBUG_ENABLED()) {
            log_debug("decode: Non-integer token at position %zd", i);
        }
        
        free(token_ids);
        PyErr_SetString(PyExc_TypeError, "All tokens must be integers");
        return NULL;
    }

    if (DEBUG_ENABLED()) {
        log_debug("decode: All tokens valid. Total text length will be %zu bytes", total_len);
    }

    char *text = (char *)malloc(total_len + 1);
    if (unlikely(!text)) {
        if (DEBUG_ENABLED()) {
            log_debug("decode: Memory allocation failed for text buffer");
        }
        free(token_ids);
        return PyErr_NoMemory();
    }

    // Second pass: copy strings with single-pass concatenation
    if (DEBUG_ENABLED()) {
        log_debug("decode: Pass 2 - Concatenating token texts");
    }
    
    size_t offset = 0;
    for (Py_ssize_t i = 0; i < token_num; i++) {
        const char *word = vocab_decode[token_ids[i]];
        size_t word_len = strlen(word);
        
        if (DEBUG_ENABLED() && (i < 3 || i == token_num - 1)) {
            log_debug("decode: Token[%zd]=ID:%d, text='%s', len=%zu", 
                     i, token_ids[i], word, word_len);
        }
        
        memcpy(text + offset, word, word_len);
        offset += word_len;
    }
    text[offset] = '\0';
    
    if (DEBUG_ENABLED()) {
        log_debug("decode: Final text length: %zu bytes", offset);
        
        if (offset <= 100) {
            log_debug("decode: Result text: '%s'", text);
        } else {
            log_debug("decode: Result text (first 100 chars): '%.100s...'", text);
        }
    }
    
    free(token_ids);
    PyObject *result = PyUnicode_FromString(text);
    free(text);
    
    if (DEBUG_ENABLED()) {
        log_debug("decode: Completed successfully");
    }
    
    return result;
}