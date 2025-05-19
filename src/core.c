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

void bpe_encode(struct HashMap *vocab, Boundary token_boundaries[], int tokens[], int *token_num)
{
    while (1)
    {
        int min_idx = -1;
        int min_rank = -1;

        for (int i = 0; i < *token_num - 1; i++)
        {
            const uint8_t *s1 = token_boundaries[i].start;
            const uint8_t *e1 = token_boundaries[i].end;
            int l1 = (int)(e1 - s1) + 1;

            const uint8_t *s2 = token_boundaries[i + 1].start;
            const uint8_t *e2 = token_boundaries[i + 1].end;
            int l2 = (int)(e2 - s2) + 1;

            int len = l1 + l2;
            char pair[len + 1];
            memcpy(pair, s1, l1);
            memcpy(pair + l1, s2, l2);
            pair[len] = '\0';

            struct Token probe = { .key = pair, .value = 0 };
            int rank = hashmap_get(vocab, &probe);

            if (rank != -1 && (min_rank == -1 || rank < min_rank))
            {
                min_idx = i;
                min_rank = rank;
            }
        }

        if (min_rank == -1)
            break;
        assert(min_idx != -1);

        token_boundaries[min_idx].end = token_boundaries[min_idx + 1].end;

        for (int i = min_idx + 1; i < *token_num - 1; i++)
        {
            token_boundaries[i].start = token_boundaries[i + 1].start;
            token_boundaries[i].end = token_boundaries[i + 1].end;
        }
        *token_num -= 1;
    }

    for (int i = 0; i < *token_num; i++)
    {
        const uint8_t *start = token_boundaries[i].start;
        const uint8_t *end = token_boundaries[i].end;
        int len = (int)(end - start) + 1;

        char tmp[len + 1];
        memcpy(tmp, start, len);
        tmp[len] = '\0';

        struct Token probe = { .key = tmp, .value = 0 };
        int rank = hashmap_get(vocab, &probe);
        
        // Tokens that get -1 rank
        if (rank < 0) {
            log_debug("WARNING: Token '%s' not found in vocabulary! Assigned rank %d", tmp, rank);
        }
        
        tokens[i] = rank;
    }
}

void encode(const uint8_t *text, struct HashMap *vocab, regex_t *regex, int tokens[], int *tokens_size) {
    log_debug("Starting encode function with text: %.64s", text);

    regmatch_t match;
    const uint8_t *cursor = text;

    #define MAX_WORD_LEN 1024
    Boundary stack_boundaries[MAX_WORD_LEN];
    int stack_tokens[MAX_WORD_LEN];

    *tokens_size = 0;

    while (regexec(regex, (const char *)cursor, 1, &match, 0) == 0) {
        int word_start = match.rm_so;
        int word_end = match.rm_eo;
        int word_len = word_end - word_start;

        log_debug("Matched word: start=%d, end=%d, length=%d", word_start, word_end, word_len);

        if (word_len <= 0) {
            log_debug("Zero or negative word length, skipping...");
            cursor += word_end;
            continue;
        }

        Boundary *word_token_boundaries = word_len <= MAX_WORD_LEN ? stack_boundaries : malloc(word_len * sizeof(Boundary));
        int *word_tokens = word_len <= MAX_WORD_LEN ? stack_tokens : malloc(word_len * sizeof(int));

        // Set up boundaries for each byte in the word
        for (int i = 0; i < word_len; i++) {
            word_token_boundaries[i].start = cursor + word_start + i;
            word_token_boundaries[i].end = cursor + word_start + i;
        }

        int word_token_num = word_len;
        bpe_encode(vocab, word_token_boundaries, word_tokens, &word_token_num);

       // Only extract and log token text when debugging is enabled
        if (DEBUG_ENABLED()) {
            for (int i = 0; i < word_token_num; i++) {
                // Get the actual token text for debugging
                const uint8_t *start = word_token_boundaries[i].start;
                const uint8_t *end = word_token_boundaries[i].end;
                int len = (int)(end - start) + 1;
                
                char token_text[len + 1];
                memcpy(token_text, start, len);
                token_text[len] = '\0';
                
                log_debug("Token #%d: ID=%d, Text='%s'", *tokens_size + i, word_tokens[i], token_text);
            }
        }

        // Copy tokens for this word into the output array
        memcpy(tokens + *tokens_size, word_tokens, word_token_num * sizeof(int));
        *tokens_size += word_token_num;

        if (word_token_boundaries != stack_boundaries) free(word_token_boundaries);
        if (word_tokens != stack_tokens) free(word_tokens);

        cursor += word_end;
    }

    log_debug("Completed encode function. Total tokens: %d", *tokens_size);
}

PyObject *decode(PyObject *tokens, char **vocab_decode, int vocab_size)
{
    log_debug("Entered decode function");

    Py_ssize_t token_num = PyList_Size(tokens);
    log_debug("Number of tokens to decode: %zd", token_num);

    size_t total_len = 0;
    int *token_ids = malloc(token_num * sizeof(int));
    for (Py_ssize_t i = 0; i < token_num; i++) {
        PyObject *token = PyList_GetItem(tokens, i);
        if (!PyLong_Check(token)) {
            log_debug("Error: Token at index %zd is not an integer", i);
            PyErr_SetString(PyExc_TypeError, "All elements of the list must be integers");
            free(token_ids);
            return NULL;
        }
        int item = (int)PyLong_AsLong(token);
        if (item < 0 || item >= vocab_size) {
            log_debug("Error: Token value %d at index %zd is out of bounds (vocab_size = %d)", 
                     item, i, vocab_size);
            PyErr_SetString(PyExc_ValueError, "Element must be non-negative and less than vocab size.");
            free(token_ids);
            return NULL;
        }
        token_ids[i] = item;
        total_len += strlen(vocab_decode[item]);
    }

    uint8_t *text = (uint8_t *)malloc(total_len + 1);
    if (!text) {
        log_debug("Error: Memory allocation failed for text buffer");
        free(token_ids);
        return PyErr_NoMemory();
    }

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
