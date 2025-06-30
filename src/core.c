#include "hutoken/core.h"

#include "Python.h"
#include "fomalib.h"
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
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <time.h>

#include "hutoken/hashmap.h"
#include "hutoken/helper.h"

void bpe_encode(struct HashMap* vocab,
                struct Boundary token_boundaries[],
                int tokens[],
                int* token_num) {
    while (true) {
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

            (void)strncpy(pair, s1, l1);
            (void)strncpy(pair + l1, s2, l2);
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
        (void)strncpy(string, start, len);
        string[len] = '\0';

        int rank = hashmap_get(vocab, &(struct Token){.key = string});

        tokens[i] = rank;
    }
}

void encode(char *text, struct HashMap *vocab, char *pattern, int tokens[], int *tokens_size) {
    
void encode(char* text,
            struct HashMap* vocab,
            char* pattern,
            int tokens[],
            int* tokens_size) {
    log_debug("Starting encode function with text: %s", text);

    PCRE2_SPTR search_term = (PCRE2_SPTR) pattern;
    PCRE2_SPTR subject = (PCRE2_SPTR) text;

    pcre2_code *regex = pcre2_compile(
        (PCRE2_SPTR) pattern,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF,
        &error_number,
        &error_offset,
        NULL);
    
    if(regex == NULL){
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(error_number,buffer,sizeof(buffer));
        log_debug("PCRE2 compilation failed at offset %d: %s\n",(int)error_offset, buffer);
        PyErr_Format(PyExc_RuntimeError, "Regex could not be compiled.");
        return;
    }

    re = pcre2_compile(pattern,PCRE2_ZERO_TERMINATED,0,&errornumber,&errorofset, NULL);

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

    while(start_offset < subject_length){
        int rc = pcre2_match(
            regex,
            subject,
            subject_length,
            start_offset,
            0,
            match_data,
            NULL);

        if(rc < 0){
            if(rc == PCRE2_ERROR_NOMATCH){
                break;
            }
        }
        
        PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);

        PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE match_start = ovector[0];
        PCRE2_SIZE match_end = ovector[1];
        PCRE2_SIZE word_len = match_end - match_start;

        log_debug("Matched word: start=%d, end=%d, length=%d", (int)match_start, (int)match_end, (int)word_len);

        int i = 0;

        Boundary word_token_boundaries[word_len];

        for(char *ptr = text + match_start; ptr < text + match_end; ptr++){
            char *start = ptr;
            char *end = ptr;
            Boundary word_token_boundary = {start, end};
            word_token_boundaries[i] = word_token_boundary;
            i++;
        }

        int word_token_num = i;
        int word_tokens[word_len];

        bpe_encode(vocab, word_token_boundaries, word_tokens, &word_token_num);

        for(int i = 0; i < word_token_num; i++){
            tokens[i + *tokens_size] = word_tokens[i];
            log_debug("Encoded token: %d", word_tokens[i]);
        }

        *tokens_size += word_token_num;

        start_offset = match_end;

        if(match_start == match_end){
            if(start_offset >= subject_length){
                break;
            }
            start_offset++;
        }

    }

    pcre2_match_data_free(match_data);
    pcre2_code_free(regex);
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

        (void)strcat(text, word);
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

        strncpy(tmp, split_morphemes, tmp_len - 1);
        tmp[tmp_len - 1] = '\0';

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
