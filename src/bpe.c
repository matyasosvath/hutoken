#include "hutoken/bpe.h"

#include "Python.h"

#include <assert.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/hashmap.h"
#include "hutoken/helper.h"

void create_words(char* text,
                  const char* pattern,
                  struct Boundary token_boundaries[],
                  size_t token_num) {
    int errorcode;
    PCRE2_SIZE erroroffset;
    pcre2_code *re;
    pcre2_match_data *match_data;
    PCRE2_SIZE *ovector;

    // Compile the pattern
    re = pcre2_compile(
        (PCRE2_SPTR)pattern,       /* the pattern */
        PCRE2_ZERO_TERMINATED,     /* indicates pattern is zero-terminated */
        PCRE2_UTF,                 /* using UTF-8 */
        &errorcode,                /* for error code */
        &erroroffset,              /* for error offset */
        NULL);                     /* use default compile context */

    if (re == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errorcode, buffer, sizeof(buffer));
        fprintf(stderr, "PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, buffer);
        PyErr_SetString(PyExc_RuntimeError, "Regex could not be compiled.");
        return;
    }

    // Create match data block
    match_data = pcre2_match_data_create_from_pattern(re, NULL);
    if (match_data == NULL) {
        pcre2_code_free(re);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create match data.");
        return;
    }

    char* cursor = text;
    size_t text_len = strlen(text);
    int i = 0;
    PCRE2_SIZE start_offset = 0;

    while (start_offset < text_len) {
        int rc = pcre2_match(
            re,                    /* the compiled pattern */
            (PCRE2_SPTR)text,     /* the subject string */
            text_len,             /* the length of the subject */
            start_offset,         /* start at this offset */
            0,                    /* default options */
            match_data,           /* block for storing the result */
            NULL);                /* use default match context */
        
        if (rc < 0) {
            // No more matches
            break;
        }
        
        ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE match_start = ovector[0];
        PCRE2_SIZE match_end = ovector[1];
        
        // Create token boundaries for each character in the match
        for (PCRE2_SIZE pos = match_start; pos < match_end; pos++) {
            char* start = text + pos;
            char* end = text + pos;
            
            struct Boundary token_boundary = {.start = start, .end = end};
            token_boundaries[i] = token_boundary;
            i++;
        }
        
        // Move to the next position after this match
        start_offset = match_end;
        
        // Handle zero-length matches
        if (match_start == match_end) {
            if (start_offset >= text_len) {
                break;
            }
            start_offset++;
        }
    }
    
    pcre2_match_data_free(match_data);
    pcre2_code_free(re);
}

void bpe_train_core(struct HashMap* vocab,
                    char* text,
                    struct Boundary token_boundaries[],
                    size_t token_num,
                    size_t vocab_size) {
    size_t token_n = token_num;
    struct Token prev_common_pair = {.key = NULL, .value = -1};
    struct Token most_common_pair = {.key = NULL, .value = -1};

    while (vocab->count < vocab_size) {
        struct HashMap* stats = hashmap_new(token_n);

        // This check prevents undefined behavior. If the number of tokens is
        // reduced to < 2, the next loop iteration would attempt to declare
        // a zero-sized VLA, which is illegal in C.
        if (token_n < 2) {
            break;
        }

        // find most common pair -> next token

        for (size_t i = 0; i < token_num - 1; i++) {
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

            int freq = hashmap_get(stats, &(struct Token){.key = pair});
            if (freq != 0) {
                hashmap_set(stats, &(struct Token){.key = strdup(pair),
                                                   .value = ++freq});
            } else {
                int initial_freq = 1;
                hashmap_set(stats, &(struct Token){.key = strdup(pair),
                                                   .value = initial_freq});
            }

            int rank = hashmap_get(stats, &(struct Token){.key = pair});

            if (most_common_pair.value < rank) {
                if (most_common_pair.key != NULL) {
                    free(most_common_pair.key);
                }
                most_common_pair.value = rank;
                most_common_pair.key = strdup(pair);
            }
        }

        // WARNING: vocab->count is a size_t. If it exceeds INT_MAX, this
        // assignment will overflow, causing token to become negative. The
        // Token.value field should ideally be changed to size_t to
        // support larger vocabs.
        int token = (int)(vocab->count + 1);

        // add new token

        hashmap_set(vocab, &(struct Token){.key = most_common_pair.key,
                                           .value = token});

        // merge that most common pair in all tokens, i.e.
        // update token boundaries for the new token everywhere

        int j = 0;
        struct Boundary new_token_boundaries[token_n];

        for (size_t i = 0; i < token_n - 1; i++) {
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

            if (most_common_pair.key != NULL &&
                strcmp(pair, most_common_pair.key) == 0) {
                struct Boundary new_token_boundary = {.start = s1, .end = e2};
                new_token_boundaries[j] = new_token_boundary;
                j++;
                i++;
            } else {
                new_token_boundaries[j] = token_boundaries[i];
                j++;
            }
        }

        for (int k = 0; k < j; k++) {
            token_boundaries[k] = new_token_boundaries[k];
        }
        token_n = j;

        visualize_bpe_train(most_common_pair,
                            token_n);

        hashmap_free(stats);

        if (prev_common_pair.key != NULL &&
            strcmp(prev_common_pair.key, most_common_pair.key) == 0) {
            break;
        }

        prev_common_pair.key = most_common_pair.key;
        prev_common_pair.value = most_common_pair.value;

        most_common_pair.value = -1;
        most_common_pair.key = NULL;
    }
}

void bpe_train(char* text,
               const int vocab_size,
               const char* pattern,
               char* vocab_file_name) {
    char* k = NULL;
    struct HashMap* vocab = hashmap_new(vocab_size);

    // add tokens for each individual byte value
    for (int i = 0; i < 256; i++) {
        char key[2];
        key[0] = (char)i;  // store ascii character
        key[1] = '\0';

        k = strdup(key);
        hashmap_set(vocab, &(struct Token){.key = k, .value = i});
    }

    size_t token_num = strlen(text);
    struct Boundary token_boundaries[token_num];

    create_words(text, pattern, token_boundaries, token_num);

    bpe_train_core(vocab, text, token_boundaries, token_num, vocab_size);

    save_vocab(vocab, vocab_file_name);

    hashmap_free(vocab);
    free(k);
}
