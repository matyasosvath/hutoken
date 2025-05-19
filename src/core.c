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
    // Pre-allocate buffer for pair string to avoid repeated allocations
    char pair_buffer[4096]; // Large enough for most common pairs
    
    // Use a flag to track if any merges were made in this pass
    bool merged;
    
    do {
        merged = false;
        int min_idx = -1;
        int min_rank = -1;

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
            
            // Use pre-allocated buffer instead of stack allocation in loop
            memcpy(pair_buffer, s1, l1);
            memcpy(pair_buffer + l1, s2, l2);
            pair_buffer[len] = '\0';

            struct Token probe = { .key = pair_buffer, .value = 0 };
            int rank = hashmap_get(vocab, &probe);

            if (rank != -1 && (min_rank == -1 || rank < min_rank)) {
                min_idx = i;
                min_rank = rank;
            }
        }

        // If we found a pair to merge
        if (min_rank != -1) {
            merged = true;
            token_boundaries[min_idx].end = token_boundaries[min_idx + 1].end;

            // Use memmove for overlapping regions instead of copying one by one
            memmove(&token_boundaries[min_idx + 1], 
                    &token_boundaries[min_idx + 2],
                    (*token_num - min_idx - 2) * sizeof(Boundary));
            
            (*token_num)--;
        }
    } while (merged);

    // Process tokens in a single pass
    for (int i = 0; i < *token_num; i++) {
        const uint8_t *start = token_boundaries[i].start;
        const uint8_t *end = token_boundaries[i].end;
        int len = (int)(end - start) + 1;

        if (len >= 4096) {
            tokens[i] = -1; // Handle oversized token
            continue;
        }

        memcpy(pair_buffer, start, len);
        pair_buffer[len] = '\0';

        struct Token probe = { .key = pair_buffer, .value = 0 };
        tokens[i] = hashmap_get(vocab, &probe);
    }
}

void encode(const uint8_t *text, struct HashMap *vocab, regex_t *regex, int tokens[], int *tokens_size) {
    // Skip debug logging in hot path
    
    regmatch_t match;
    const uint8_t *cursor = text;
    const uint8_t *end = text + strlen((const char*)text);

    // Pre-allocate fixed arrays for better cache locality
    #define MAX_WORD_LEN 1024
    Boundary stack_boundaries[MAX_WORD_LEN];
    int stack_tokens[MAX_WORD_LEN];

    *tokens_size = 0;
    int max_tokens = 65536; // Assume this is the max tokens array size

    // Fast path for common case
    while (cursor < end && *tokens_size < max_tokens) {
        if (regexec(regex, (const char *)cursor, 1, &match, 0) != 0) 
            break;
            
        int word_start = match.rm_so;
        int word_end = match.rm_eo;
        int word_len = word_end - word_start;

        if (word_len <= 0) {
            cursor += (word_end > 0) ? word_end : 1;
            continue;
        }

        // Use stack allocation for most common case
        Boundary *word_token_boundaries = stack_boundaries;
        int *word_tokens = stack_tokens;
        
        // Only allocate heap memory for unusually long words
        if (word_len > MAX_WORD_LEN) {
            word_token_boundaries = malloc(word_len * sizeof(Boundary));
            word_tokens = malloc(word_len * sizeof(int));
            if (!word_token_boundaries || !word_tokens) {
                if (word_token_boundaries) free(word_token_boundaries);
                if (word_tokens) free(word_tokens);
                return; // Handle allocation failure
            }
        }

        // Use memset for faster initialization or unroll small loops
        for (int i = 0; i < word_len; i++) {
            word_token_boundaries[i].start = cursor + word_start + i;
            word_token_boundaries[i].end = cursor + word_start + i;
        }

        int word_token_num = word_len;
        bpe_encode(vocab, word_token_boundaries, word_tokens, &word_token_num);

        // Only do detailed logging when debug is enabled AND only once per batch
        if (DEBUG_ENABLED() && (*tokens_size == 0 || *tokens_size % 100 == 0)) {
            // Logging code here
        }

        // Check if we have enough space in tokens array
        if (*tokens_size + word_token_num > max_tokens) {
            word_token_num = max_tokens - *tokens_size;
        }

        // Copy tokens for this word into the output array
        memcpy(tokens + *tokens_size, word_tokens, word_token_num * sizeof(int));
        *tokens_size += word_token_num;

        if (word_token_boundaries != stack_boundaries) free(word_token_boundaries);
        if (word_tokens != stack_tokens) free(word_tokens);

        cursor += word_end;
    }
}

PyObject *decode(PyObject *tokens, char **vocab_decode, int vocab_size)
{
    log_debug("Entered decode function");

    Py_ssize_t token_num = PyList_Size(tokens);
    log_debug("Number of tokens to decode: %zd", token_num);

    // Allocate a single block for both token IDs and resulting text
    // Pre-calculate total string length to avoid reallocation
    size_t total_len = 0;
    int *token_ids = malloc(token_num * sizeof(int));
    
    // First pass - validate tokens and calculate total length
    for (Py_ssize_t i = 0; i < token_num; i++) {
        PyObject *token = PyList_GetItem(tokens, i);
        int item = (int)PyLong_AsLong(token);
        
        // Fast path for typical case - skip repeated checks
        if (likely(item >= 0 && item < vocab_size)) {
            token_ids[i] = item;
            total_len += strlen(vocab_decode[item]);
            continue;
        }
        
        // Error handling for less common cases
        if (!PyLong_Check(token)) {
            free(token_ids);
            PyErr_SetString(PyExc_TypeError, "All elements of the list must be integers");
            return NULL;
        }
        
        free(token_ids);
        PyErr_SetString(PyExc_ValueError, "Element must be non-negative and less than vocab size.");
        return NULL;
    }

    // One allocation with exact size
    uint8_t *text = (uint8_t *)malloc(total_len + 1);
    if (!text) {
        free(token_ids);
        return PyErr_NoMemory();
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

    PyObject *result = PyUnicode_FromString((const char *)text);
    free(text);
    return result;
}
