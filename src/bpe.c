#include "hutoken/bpe.h"

#include "Python.h"

#include <assert.h>
#include <regex.h>
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
    regex_t regex;

    int r = regcomp(&regex, pattern, REG_EXTENDED);
    if (r) {
        (void)fputs("Regex could not be compiled.", stderr);
        PyErr_SetString(PyExc_RuntimeError, "Regex could not be compiled.");
        return;
    }

    regmatch_t match;
    char* cursor = text;
    int i = 0;

    while (regexec(&regex, cursor, 1, &match, 0) == 0) {
        int word_start = match.rm_so;
        int word_end = match.rm_eo;

        for (char* ptr = cursor + word_start; ptr < cursor + word_end; ptr++) {
            char* start = ptr;
            char* end = ptr;

            struct Boundary token_boundary = {start, end};

            token_boundaries[i] = token_boundary;
            i += 1;
        }
        cursor += word_end;
    }

    regfree(&regex);
}

void bpe_train_core(struct HashMap* vocab,
                    char* text,
                    struct Boundary token_boundaries[],
                    size_t token_num,
                    size_t vocab_size) {
    size_t token_n = token_num;
    struct Token prev_common_pair = {"", -1};
    struct Token most_common_pair = {"", -1};

    while (vocab->count < vocab_size) {
        struct HashMap* stats = hashmap_new(token_n);

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
            if (freq) {
                hashmap_set(stats, &(struct Token){.key = strdup(pair),
                                                   .value = ++freq});
            } else {
                int initial_freq = 1;
                hashmap_set(stats, &(struct Token){.key = strdup(pair),
                                                   .value = initial_freq});
            }

            int rank = hashmap_get(stats, &(struct Token){.key = pair});

            if (most_common_pair.value < rank) {
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

            if (strcmp(pair, most_common_pair.key) == 0) {
                struct Boundary new_token_boundary = {s1, e2};
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

        visualize_bpe_train(text, token_boundaries, most_common_pair, token,
                            token_n);

        hashmap_free(stats);

        if (strcmp(prev_common_pair.key, most_common_pair.key) == 0) {
            break;
        }

        prev_common_pair.key = most_common_pair.key;
        prev_common_pair.value = most_common_pair.value;

        most_common_pair.value = -1;
        most_common_pair.key = "\0";
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
