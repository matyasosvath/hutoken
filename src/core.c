#include "hutoken/core.h"

#include <Python.h>

#include <assert.h>
#include <regex.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <time.h>

#include "hutoken/hashmap.h"
#include "hutoken/helper.h"

void bpe_encode(struct HashMap* vocab,
                struct Boundary token_boundaries[],
                int tokens[],
                int* token_num) {
    while (1) {
        int min_idx = -1;
        int min_rank = -1;

        for (int i = 0; i < *token_num - 1; i++) {
            char* s1 = token_boundaries[i].start;
            char* e1 = token_boundaries[i].end;
            ptrdiff_t l1 = (e1 - s1) + 1;

            char* s2 = token_boundaries[i + 1].start;
            char* e2 = token_boundaries[i + 1].end;
            ptrdiff_t l2 = (e2 - s2) + 1;

            ptrdiff_t len = l1 + l2;
            char pair[len + 1];

            strncpy(pair, s1, l1);
            strncpy(pair + l1, s2, l2);
            pair[len] = '\0';

            int rank = hashmap_get(vocab, &(struct Token){.key = pair});

            if (rank != -1 && (min_rank == -1 || rank < min_rank)) {
                min_idx = i;
                min_rank = rank;
            }
        }

        // no pairs to merge
        if (min_rank == -1) {
            break;
        }

        assert(min_idx != -1);

        // merge pairs, leave rest unchanged
        token_boundaries[min_idx].end = token_boundaries[min_idx + 1].end;

        for (int i = min_idx + 1; i < *token_num - 1; i++) {
            token_boundaries[i].start = token_boundaries[i + 1].start;
            token_boundaries[i].end = token_boundaries[i + 1].end;
        }
        *token_num -= 1;
    }

    // update tokens
    for (int i = 0; i < *token_num; i++) {
        char* start = token_boundaries[i].start;
        char* end = token_boundaries[i].end;
        ptrdiff_t len = (end - start) + 1;

        char string[len + 1];
        strncpy(string, start, len);
        string[len] = '\0';

        int rank = hashmap_get(vocab, &(struct Token){.key = string});

        tokens[i] = rank;
    }
}

void encode(char* text,
            struct HashMap* vocab,
            char* pattern,
            int tokens[],
            int* tokens_size) {
    log_debug("Starting encode function with text: %s", text);

    PCRE2_SPTR search_term = (PCRE2_SPTR) pattern;
    PCRE2_SPTR subject = (PCRE2_SPTR) text;

    pcre2_code *re;
    pcre2_match_data *match_data;
    int errornumber;
    PCRE2_SIZE errorofset;
    int rc;

    re = pcre2_compile(search_term,PCRE2_ZERO_TERMINATED,0,&errornumber,&errorofset, NULL);

    if (re = NULL)
    {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errornumber,buffer,sizeof(buffer));
        log_debug("PCRE2 compilation failed at offset %d: %s\n",(int)errorofset, buffer);
        PyErr_SetString(PyExc_RuntimeError, "Regex could not be compiled.");
        return;
    }
    
    match_data = pcre2_match_data_create_from_pattern(re,NULL);

    PCRE2_SIZE start_offset = 0;
    int match_count = 0;

    // regex_t regex;
    // int r = regcomp(&regex, pattern, REG_EXTENDED);
    // if (r) {
    //    log_debug("Error: Regex could not be compiled.");
    //    PyErr_SetString(PyExc_RuntimeError, "Regex could not be compiled.");
    //    return;
    // }

    // regmatch_t match;
    
    char *cursor = text;

    while(1){
        rc = pcre2_match(re, subject, strlen((char*)subject),
        start_offset, 0, match_data, NULL);

        if (rc < 0)
        {
            if (rc == PCRE2_ERROR_NOMATCH)
            {
                break;
            }
            else
            {
                printf("Matching error %d\n", rc);
                break;
            }
        }

        match_count++;
        
        PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);

        //while (regexec(&regex, cursor, 1, &match, 0) == 0) {

        //int word_start = match.rm_so;
        //int word_start = (int) ovector[0];
        PCRE2_SIZE word_start = ovector[0];
        //int word_end = match.rm_eo;
        PCRE2_SIZE word_end = ovector[1];
        //int word_end = (int) ovector[1];

        int word_len = word_end - word_start;

        log_debug("Matched word: start=%d, end=%d, length=%d", word_start,
                  word_end, word_len);

        int i = 0;
        struct Boundary word_token_boundaries[word_len];

        for (char* ptr = cursor + word_start; ptr < cursor + word_end; ptr++) {
            char* start = ptr;
            char* end = ptr;
            struct Boundary word_token_boundary = {start, end};
            word_token_boundaries[i] = word_token_boundary;
            i++;
        }

        int word_token_num = word_len;
        int word_tokens[word_len];

        bpe_encode(vocab, word_token_boundaries, word_tokens, &word_token_num);

        for (int i = 0; i < word_token_num; i++) {
            tokens[i + *tokens_size] = word_tokens[i];
            log_debug("Encoded token: %d", word_tokens[i]);
        }

        cursor += word_end;
        *tokens_size += word_token_num;
    }

    //regfree(&regex);
    pcre2_match_data_free(match_data);
    pcre2_code_free(re);
    log_debug("Completed encode function. Total tokens: %d", *tokens_size);
}

PyObject* decode(PyObject* tokens, char** vocab_decode, int vocab_size) {
    log_debug("Entered decode function");

    Py_ssize_t token_num = PyList_Size(tokens);
    log_debug("Number of tokens to decode: %zd", token_num);

    size_t text_size = sizeof(char) * ((int)token_num + 1);
    char* text = (char*)malloc(text_size);

    if (!text) {
        log_debug("Error: Memory allocation failed for text buffer");
        return PyErr_NoMemory();
    }

    text[0] = '\0';
    log_debug("Initialized text buffer to an empty string (size: %zu bytes)",
              text_size);

    for (Py_ssize_t i = 0; i < token_num; i++) {
        log_debug("Processing token at index %zd", i);

        PyObject* token = PyList_GetItem(tokens, i);
        if (!PyLong_Check(token)) {
            log_debug("Error: Token at index %zd is not an integer", i);
            PyErr_SetString(PyExc_TypeError,
                            "All elements of the list must be integers");
            free(text);
            return NULL;
        }

        int item = (int)PyLong_AsLong(token);
        if (item < 0 || item >= vocab_size) {
            log_debug(
                "Error: Token value %d is out of bounds (vocab_size = %d)",
                item, vocab_size);
            PyErr_SetString(
                PyExc_ValueError,
                "Element must be non-negative and less than vocab size.");
            free(text);
            return NULL;
        }

        const char* word = vocab_decode[item];
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
                return PyErr_NoMemory();
            }
            text = new_text;
            text_size = buffer_size;

            log_debug("Resized text buffer to new size: %d bytes", buffer_size);
        }

        strcat(text, word);
        log_debug(
            "Appended word '%s' to text buffer. Current text: '%s' (buffer "
            "size: %zu bytes)",
            word, text, text_size);
    }

    PyObject* result = PyUnicode_FromString(text);
    if (!result) {
        log_debug("Error: Failed to create Python string from decoded text");
        free(text);
        return NULL;
    }

    log_debug("Successfully created Python string from decoded text: '%s'",
              text);

    free(text);
    return result;
}
