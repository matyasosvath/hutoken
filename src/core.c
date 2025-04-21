#include <Python.h>
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
            char *s1 = token_boundaries[i].start;
            char *e1 = token_boundaries[i].end;
            int l1 = (e1 - s1) + 1;

            char *s2 = token_boundaries[i + 1].start;
            char *e2 = token_boundaries[i + 1].end;
            int l2 = (e2 - s2) + 1;

            int len = l1 + l2;
            char pair[len + 1];

            strncpy(pair, s1, l1);
            strncpy(pair + l1, s2, l2);
            pair[len] = '\0';

            int rank = hashmap_get(vocab, &(struct Token){.key = pair});

            if (rank != -1 && (min_rank == -1 || rank < min_rank))
            {
                min_idx = i;
                min_rank = rank;
            }
        }

        // no pairs to merge
        if (min_rank == -1)
            break;
        assert(min_idx != -1);

        // merge pairs, leave rest unchanged
        token_boundaries[min_idx].end = token_boundaries[min_idx + 1].end;

        for (int i = min_idx + 1; i < *token_num - 1; i++)
        {
            token_boundaries[i].start = token_boundaries[i + 1].start;
            token_boundaries[i].end = token_boundaries[i + 1].end;
        }
        *token_num -= 1;
    }

    // update tokens
    for (int i = 0; i < *token_num; i++)
    {
        char *start = token_boundaries[i].start;
        char *end = token_boundaries[i].end;
        int len = (end - start) + 1;

        char string[len + 1];
        strncpy(string, start, len);
        string[len] = '\0';

        int rank = hashmap_get(vocab, &(struct Token){.key = string});

        tokens[i] = rank;
    }
}

// Optimized encode: regex is precompiled and passed as a pointer
void encode(char *text, struct HashMap *vocab, regex_t *regex, int tokens[], int *tokens_size) {
    log_debug("Starting encode function with text: %s", text);

    regmatch_t match;
    char *cursor = text;

    while (regexec(regex, cursor, 1, &match, 0) == 0) {
        int word_start = match.rm_so;
        int word_end = match.rm_eo;
        int word_len = word_end - word_start;
        log_debug("Matched word: start=%d, end=%d, length=%d, text='%.*s'", word_start, word_end, word_len, word_len, cursor + word_start);

        if (word_len <= 0) {
            log_debug("Zero or negative word length, skipping...");
            cursor += word_end;
            continue;
        }

        Boundary stack_boundaries[256];
        Boundary *word_token_boundaries = word_len <= 256 ? stack_boundaries : malloc(word_len * sizeof(Boundary));
        int i = 0;
        for (char *ptr = cursor + word_start; ptr < cursor + word_end; ptr++, i++) {
            word_token_boundaries[i].start = ptr;
            word_token_boundaries[i].end = ptr;
        }

        int word_token_num = word_len;
        int stack_tokens[256];
        int *word_tokens = word_len <= 256 ? stack_tokens : malloc(word_len * sizeof(int));

        bpe_encode(vocab, word_token_boundaries, word_tokens, &word_token_num);

        // Log every token with its value and string representation
        for (int j = 0; j < word_token_num; j++) {
            // Reconstruct the string for this token
            char *start = word_token_boundaries[j].start;
            char *end = word_token_boundaries[j].end;
            int len = (end - start) + 1;
            char token_str[len + 1];
            strncpy(token_str, start, len);
            token_str[len] = '\0';

            log_debug("Encoded token: '%s' -> %d", token_str, word_tokens[j]);
            if (word_tokens[j] < 0) {
                log_debug("Warning: Unknown token emitted for '%s'", token_str);
            }
        }

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

    // 1. Precompute total output length
    size_t total_len = 0;
    for (Py_ssize_t i = 0; i < token_num; i++) {
        PyObject *token = PyList_GetItem(tokens, i);
        if (!PyLong_Check(token)) {
            log_debug("Error: Token at index %zd is not an integer", i);
            PyErr_SetString(PyExc_TypeError, "All elements of the list must be integers");
            return NULL;
        }
        int item = (int)PyLong_AsLong(token);
        if (item < 0 || item >= vocab_size) {
            log_debug("Error: Token value %d is out of bounds (vocab_size = %d)", item, vocab_size);
            PyErr_SetString(PyExc_ValueError, "Element must be non-negative and less than vocab size.");
            return NULL;
        }
        total_len += strlen(vocab_decode[item]);
    }

    // 2. Allocate buffer once
    size_t text_size = total_len + 1;
    char *text = (char *)malloc(text_size);
    if (!text) {
        log_debug("Error: Memory allocation failed for text buffer");
        return PyErr_NoMemory();
    }
    text[0] = '\0';
    log_debug("Initialized text buffer to an empty string (size: %zu bytes)", text_size);

    // 3. Copy words directly
    size_t offset = 0;
    for (Py_ssize_t i = 0; i < token_num; i++) {
        log_debug("Processing token at index %zd", i);

        PyObject *token = PyList_GetItem(tokens, i);
        int item = (int)PyLong_AsLong(token);
        const char *word = vocab_decode[item];
        size_t word_len = strlen(word);
        log_debug("Decoded token value %d to word '%s' (length: %zu)", item, word, word_len);

        memcpy(text + offset, word, word_len);
        offset += word_len;
        text[offset] = '\0';

        log_debug("Appended word '%s' to text buffer. Current text: '%s' (buffer size: %zu bytes)", word, text, text_size);
    }

    PyObject *result = PyUnicode_FromString(text);
    if (!result) {
        log_debug("Error: Failed to create Python string from decoded text");
        free(text);
        return NULL;
    }

    log_debug("Successfully created Python string from decoded text: '%s'", text);

    free(text);
    return result;
}
