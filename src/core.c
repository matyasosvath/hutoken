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

#include "helper.c"
#include "hashmap.c"

#define MAX_CACHE_SIZE 1024
static struct {
    char *key;
    int *tokens;
    int token_count;
} token_cache[MAX_CACHE_SIZE];
static int cache_size = 0;

void bpe_encode(struct HashMap *vocab, Boundary token_boundaries[], int tokens[], int *token_num)
{
    if (DEBUG_ENABLED()) {
        log_debug("bpe_encode: Starting with %d tokens", *token_num);
    }
    
    char pair_buffer[4096]; 
    bool merged;
    int iteration = 0;
    
    do {
        iteration++;
        merged = false;
        int min_idx = -1;
        int min_rank = -1;

        if (DEBUG_ENABLED() && iteration <= 3) {
            log_debug("bpe_encode: BPE iteration %d with %d tokens", iteration, *token_num);
        }

        // Process all adjacent pairs in a single pass
        for (int i = 0; i < *token_num - 1; i++) {
            const uint8_t *s1 = token_boundaries[i].start;
            const uint8_t *e1 = token_boundaries[i].end;
            int l1 = (int)(e1 - s1) + 1;

            const uint8_t *s2 = token_boundaries[i + 1].start;
            const uint8_t *e2 = token_boundaries[i + 1].end;
            int l2 = (int)(e2 - s2) + 1;

            int len = l1 + l2;
            if (len >= 4096) continue; // Skip unusually long pairs
            
            memcpy(pair_buffer, s1, l1);
            memcpy(pair_buffer + l1, s2, l2);
            pair_buffer[len] = '\0';

            struct Token probe = { .key = pair_buffer, .value = 0 };
            int rank = hashmap_get(vocab, &probe);

            if (rank != -1 && (min_rank == -1 || rank < min_rank)) {
                min_idx = i;
                min_rank = rank;
                
                if (DEBUG_ENABLED() && i < 5) {
                    log_debug("bpe_encode: Found merge candidate at idx=%d, pair='%s', rank=%d", 
                             i, pair_buffer, rank);
                }
            }
        }

        // If we found a pair to merge
        if (min_rank != -1) {
            merged = true;
            token_boundaries[min_idx].end = token_boundaries[min_idx + 1].end;

            if (DEBUG_ENABLED() && iteration <= 3) {
                int merged_len = (int)(token_boundaries[min_idx].end - token_boundaries[min_idx].start) + 1;
                char merged_token[merged_len + 1];
                memcpy(merged_token, token_boundaries[min_idx].start, merged_len);
                merged_token[merged_len] = '\0';
                log_debug("bpe_encode: Merging at idx=%d to form '%s' with rank=%d", 
                         min_idx, merged_token, min_rank);
            }

            memmove(&token_boundaries[min_idx + 1], 
                    &token_boundaries[min_idx + 2],
                    (*token_num - min_idx - 2) * sizeof(Boundary));
            
            (*token_num)--;
        }
    } while (merged);

    if (DEBUG_ENABLED()) {
        log_debug("bpe_encode: Completed after %d iterations, final token count: %d", 
                iteration, *token_num);
    }

    // Process tokens in a single pass
    for (int i = 0; i < *token_num; i++) {
        const uint8_t *start = token_boundaries[i].start;
        const uint8_t *end = token_boundaries[i].end;
        int len = (int)(end - start) + 1;

        if (len >= 4096) {
            tokens[i] = -1; // Handle oversized token
            if (DEBUG_ENABLED()) {
                log_debug("bpe_encode: WARNING - Token at idx=%d exceeds max length, assigning -1", i);
            }
            continue;
        }

        memcpy(pair_buffer, start, len);
        pair_buffer[len] = '\0';

        struct Token probe = { .key = pair_buffer, .value = 0 };
        tokens[i] = hashmap_get(vocab, &probe);
        
        if (DEBUG_ENABLED() && (i < 5 || i >= *token_num - 5 || tokens[i] < 0)) {
            log_debug("bpe_encode: Final token[%d] = '%s' -> ID=%d", 
                    i, pair_buffer, tokens[i]);
        }
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
        if (word_len > MAX_WORD_LEN) {
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

        for (int i = 0; i < word_len; i++) {
            word_token_boundaries[i].start = cursor + word_start + i;
            word_token_boundaries[i].end = cursor + word_start + i;
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

        if (*tokens_size + word_token_num > max_tokens) {
            if (DEBUG_ENABLED()) {
                log_debug("encode: WARNING - Reaching max tokens limit, truncating from %d to %d", 
                        word_token_num, max_tokens - *tokens_size);
            }
            word_token_num = max_tokens - *tokens_size;
        }

        memcpy(tokens + *tokens_size, word_tokens, word_token_num * sizeof(int));
        *tokens_size += word_token_num;

        if (word_token_boundaries != stack_boundaries) free(word_token_boundaries);
        if (word_tokens != stack_tokens) free(word_tokens);

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
