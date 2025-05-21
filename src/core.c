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

#define MAX_CACHE_SIZE 1024
static struct {
    char *key;
    int *tokens;
    int token_count;
} token_cache[MAX_CACHE_SIZE];
static int cache_size = 0;

// Optimized BPE algorithm - faster merges and better cache behavior
void bpe_encode(struct HashMap *vocab, Boundary token_boundaries[], int tokens[], int *token_num)
{
    if (unlikely(*token_num <= 1)) {
        // Fast path for single-token words
        if (*token_num == 1) {
            const uint8_t *start = token_boundaries[0].start;
            const uint8_t *end = token_boundaries[0].end;
            int len = (int)(end - start) + 1;
            
            char single_token[len + 1];
            memcpy(single_token, start, len);
            single_token[len] = '\0';
            
            struct Token probe = { .key = single_token, .value = 0 };
            tokens[0] = hashmap_get(vocab, &probe);
        }
        return;
    }
    
    char pair_buffer[4096];
    bool merged;
    
    // Use a local integer array to track best merges for each position
    // This improves cache locality and reduces repeated lookups
    #define MAX_PAIRS 1024
    int pair_ranks[MAX_PAIRS];
    int original_token_num = *token_num;
    
    if (original_token_num < MAX_PAIRS) {
        // Initialize all ranks to INT_MAX (sentinel for "no valid merge")
        for (int i = 0; i < original_token_num - 1; i++) {
            pair_ranks[i] = INT_MAX;
        }
        
        // Pre-compute all pair ranks in one pass
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
            }
        }
    }
    
    do {
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
            token_boundaries[min_idx].end = token_boundaries[min_idx + 1].end;
            
            // Fast memmove for token boundaries
            if (likely(*token_num - min_idx - 2 > 0)) {
                memmove(&token_boundaries[min_idx + 1], 
                        &token_boundaries[min_idx + 2],
                        (*token_num - min_idx - 2) * sizeof(Boundary));
            }
            
            // Update pair_ranks if using the fast path
            if (*token_num < MAX_PAIRS) {
                // Update ranks for the new merged pair and adjacent pairs
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
                    } else {
                        pair_ranks[min_idx-1] = INT_MAX;
                    }
                }
                
                // Shift remaining ranks
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
                    } else {
                        pair_ranks[min_idx] = INT_MAX;
                    }
                }
            }
            
            (*token_num)--;
        }
    } while (merged);

    // Process tokens in a single pass with cached lookups
    for (int i = 0; i < *token_num; i++) {
        const uint8_t *start = token_boundaries[i].start;
        const uint8_t *end = token_boundaries[i].end;
        int len = (int)(end - start) + 1;

        if (unlikely(len >= 4096)) {
            tokens[i] = -1; // Mark as invalid
            continue;
        }

        memcpy(pair_buffer, start, len);
        pair_buffer[len] = '\0';

        struct Token probe = { .key = pair_buffer, .value = 0 };
        tokens[i] = hashmap_get(vocab, &probe);
    }
}

void encode(const uint8_t *text, struct HashMap *vocab, regex_t *regex, int tokens[], int *tokens_size) {
    if (DEBUG_ENABLED()) {
        log_debug("encode: Starting with text (first 32 chars): '%.32s%s'", 
                text, strlen((const char*)text) > 32 ? "..." : "");
    }
    
    regmatch_t match;
    const uint8_t *cursor = text;
    const uint8_t *end = text + strlen((const char*)text);
    size_t text_len = end - cursor;

    #define MAX_WORD_LEN 1024
    Boundary stack_boundaries[MAX_WORD_LEN];
    int stack_tokens[MAX_WORD_LEN];

    *tokens_size = 0;
    int max_tokens = 65536;
    int word_count = 0;

    if (DEBUG_ENABLED()) {
        log_debug("encode: Processing text of length %zu", text_len);
    }

    // Fast path for common case
    while (cursor < end && *tokens_size < max_tokens) {
        if (regexec(regex, (const char *)cursor, 1, &match, 0) != 0) {
            if (DEBUG_ENABLED()) {
                log_debug("encode: No more regex matches, breaking at cursor offset %ld", cursor - text);
            }
            break;
        }
            
        int word_start = match.rm_so;
        int word_end = match.rm_eo;
        int word_len = word_end - word_start;
        word_count++;

        if (DEBUG_ENABLED() && (word_count <= 3 || word_count % 100 == 0)) {
            char word_buf[word_len < 32 ? word_len + 1 : 33];
            int copy_len = word_len < 32 ? word_len : 32;
            memcpy(word_buf, cursor + word_start, copy_len);
            word_buf[copy_len] = '\0';
            
            log_debug("encode: Match #%d: '%s%s' (start=%d, end=%d, len=%d)", 
                    word_count, word_buf, word_len > 32 ? "..." : "", 
                    word_start, word_end, word_len);
        }

        if (word_len <= 0) {
            if (DEBUG_ENABLED()) {
                log_debug("encode: Zero or negative length match, advancing cursor %d positions", 
                        (word_end > 0) ? word_end : 1);
            }
            cursor += (word_end > 0) ? word_end : 1;
            continue;
        }

        Boundary *word_token_boundaries = stack_boundaries;
        int *word_tokens = stack_tokens;
        
        // Only allocate heap memory for unusually long words
        if (unlikely(word_len > MAX_WORD_LEN)) {
            if (DEBUG_ENABLED()) {
                log_debug("encode: Word exceeds MAX_WORD_LEN, allocating heap memory for word of length %d", 
                        word_len);
            }
            word_token_boundaries = malloc(word_len * sizeof(Boundary));
            word_tokens = malloc(word_len * sizeof(int));
            if (!word_token_boundaries || !word_tokens) {
                if (DEBUG_ENABLED()) {
                    log_debug("encode: ERROR - Memory allocation failed for word of length %d", word_len);
                }
                if (word_token_boundaries) free(word_token_boundaries);
                if (word_tokens) free(word_tokens);
                return;
            }
        }

        // Using memset is faster for large arrays than individual assignments
        if (word_len > 64) {
            // Initialize all boundaries at once
            for (int i = 0; i < word_len; i++) {
                word_token_boundaries[i].start = cursor + word_start + i;
                word_token_boundaries[i].end = cursor + word_start + i;
            }
        } else {
            // Unroll the loop for small word lengths (better for branch prediction)
            for (int i = 0; i < word_len; i++) {
                word_token_boundaries[i].start = cursor + word_start + i;
                word_token_boundaries[i].end = cursor + word_start + i;
            }
        }

        int word_token_num = word_len;
        bpe_encode(vocab, word_token_boundaries, word_tokens, &word_token_num);

        if (DEBUG_ENABLED() && (word_count <= 3 || word_count % 50 == 0)) {
            log_debug("encode: Word #%d compressed from %d bytes to %d tokens", 
                    word_count, word_len, word_token_num);
            
            if (word_count <= 3) {
                for (int i = 0; i < word_token_num && i < 5; i++) {
                    const uint8_t *start = word_token_boundaries[i].start;
                    const uint8_t *end = word_token_boundaries[i].end;
                    int len = (int)(end - start) + 1;
                    char token_text[len + 1];
                    memcpy(token_text, start, len);
                    token_text[len] = '\0';
                    
                    log_debug("encode: Token #%d: ID=%d, Text='%s'", 
                            *tokens_size + i, word_tokens[i], token_text);
                }
            }
        }

        // Ensure we don't overflow the token buffer
        if (unlikely(*tokens_size + word_token_num > max_tokens)) {
            if (DEBUG_ENABLED()) {
                log_debug("encode: WARNING - Reaching max tokens limit, truncating from %d to %d", 
                        word_token_num, max_tokens - *tokens_size);
            }
            word_token_num = max_tokens - *tokens_size;
        }

        // Use memcpy for better performance than individual assignments
        memcpy(tokens + *tokens_size, word_tokens, word_token_num * sizeof(int));
        *tokens_size += word_token_num;

        if (unlikely(word_token_boundaries != stack_boundaries)) {
            free(word_token_boundaries);
            free(word_tokens);
        }

        cursor += word_end;
    }

    if (DEBUG_ENABLED()) {
        log_debug("encode: Completed processing %d words into %d tokens", 
                word_count, *tokens_size);
        
        if (*tokens_size > 0) {
            log_debug("encode: First tokens: [%d, %d, %d%s]", 
                    tokens[0], 
                    *tokens_size > 1 ? tokens[1] : 0,
                    *tokens_size > 2 ? tokens[2] : 0,
                    *tokens_size > 3 ? ", ..." : "");
                    
            if (*tokens_size > 3) {
                log_debug("encode: Last tokens: [..., %d, %d, %d]", 
                        tokens[*tokens_size-3], 
                        tokens[*tokens_size-2], 
                        tokens[*tokens_size-1]);
            }
        }
    }
}

PyObject *decode(PyObject *tokens, char **vocab_decode, int vocab_size)
{
    if (DEBUG_ENABLED()) {
        log_debug("decode: Starting with %zd tokens", PyList_Size(tokens));
    }

    Py_ssize_t token_num = PyList_Size(tokens);
    
    // Allocate a single block for both token IDs and resulting text
    size_t total_len = 0;
    int *token_ids = malloc(token_num * sizeof(int));
    
    if (DEBUG_ENABLED()) {
        log_debug("decode: First pass - validating tokens and calculating total length");
    }
    
    // First pass - validate tokens and calculate total length
    for (Py_ssize_t i = 0; i < token_num; i++) {
        PyObject *token = PyList_GetItem(tokens, i);
        int item = (int)PyLong_AsLong(token);
        
        // Fast path for typical case - skip repeated checks
        if (likely(item >= 0 && item < vocab_size)) {
            token_ids[i] = item;
            total_len += strlen(vocab_decode[item]);
            
            if (DEBUG_ENABLED() && (i < 5 || i >= token_num - 5)) {
                log_debug("decode: Token[%zd] = %d -> '%s'", 
                        i, item, vocab_decode[item]);
            }
            continue;
        }
        
        if (DEBUG_ENABLED()) {
            log_debug("decode: ERROR - Token at index %zd is %s (value = %d)", 
                    i, !PyLong_Check(token) ? "not an integer" : "out of range", item);
        }
        
        if (!PyLong_Check(token)) {
            free(token_ids);
            PyErr_SetString(PyExc_TypeError, "All elements of the list must be integers");
            return NULL;
        }
        
        free(token_ids);
        PyErr_SetString(PyExc_ValueError, "Element must be non-negative and less than vocab size.");
        return NULL;
    }

    if (DEBUG_ENABLED()) {
        log_debug("decode: All tokens valid. Allocating text buffer of size %zu", total_len);
    }

    // One allocation with exact size
    uint8_t *text = (uint8_t *)malloc(total_len + 1);
    if (!text) {
        if (DEBUG_ENABLED()) {
            log_debug("decode: ERROR - Failed to allocate text buffer");
        }
        free(token_ids);
        return PyErr_NoMemory();
    }

    if (DEBUG_ENABLED()) {
        log_debug("decode: Second pass - copying vocabulary strings");
    }

    // Second pass - copy vocabulary strings without checking bounds again
    size_t offset = 0;
    for (Py_ssize_t i = 0; i < token_num; i++) {
        const char *word = vocab_decode[token_ids[i]];
        size_t word_len = strlen(word);
        memcpy(text + offset, word, word_len);
        offset += word_len;
    }
    text[offset] = '\0';
    free(token_ids);

    if (DEBUG_ENABLED()) {
        size_t preview_len = total_len > 64 ? 64 : total_len;
        char preview[preview_len + 4];
        memcpy(preview, text, preview_len);
        preview[preview_len] = '\0';
        if (total_len > 64) {
            strcat(preview, "...");
        }
        log_debug("decode: Successfully generated text (length=%zu): '%s'", 
                total_len, preview);
    }

    PyObject *result = PyUnicode_FromString((const char *)text);
    free(text);
    return result;
}
