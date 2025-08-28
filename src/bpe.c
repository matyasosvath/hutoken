#include "hutoken/bpe.h"

#include "Python.h"

#include <assert.h>
#include <regex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/hash.h"
#include "hutoken/hashmap.h"
#include "hutoken/helper.h"
#include "hutoken/parser.h"

uint64_t token_hash(const void* item) {
    const struct Token* token = item;
    if (!token->key) {
        return 0;
    }
    return hashmap_murmur(token->key, strlen(token->key));
}

int token_compare(const void* a, const void* b) {
    const struct Token* ua = a;
    const struct Token* ub = b;
    return strcmp(ua->key, ub->key);
}

uint64_t pair_hash(const void* item) {
    const struct MergeRule* merge_item = item;
    uint64_t x = ((uint64_t)merge_item->left_id << 32) | merge_item->right_id;

    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    x = x ^ (x >> 31);

    return x;
}

int pair_compare(const void* lhs, const void* rhs) {
    const struct MergeRule* item_a = lhs;
    const struct MergeRule* item_b = rhs;
    return !(item_a->left_id == item_b->left_id &&
             item_a->right_id == item_b->right_id);
}

void create_words(char* text,
                  const char* pattern,
                  struct Boundary token_boundaries[],
                  size_t token_num) {
    regex_t regex;
    struct ParserState parser;

    if (pattern != NULL) {
        if (regcomp(&regex, pattern, REG_EXTENDED) == true) {
            (void)fputs("Regex could not be compiled.", stderr);
            PyErr_SetString(PyExc_RuntimeError, "Regex could not be compiled.");
            return;
        }
    } else {
        parser = parser_init(text);
    }

    regmatch_t match;
    char* cursor = text;
    int i = 0;
    struct TokenSlice word;

    while (true) {
        int word_start = 0;
        int word_end = 0;
        if (pattern != NULL) {
            if (regexec(&regex, cursor, 1, &match, 0) != 0) {
                break;
            }

            word_start = match.rm_so;
            word_end = match.rm_eo;
        } else {
            if (parser_next_token(&parser, &word) == false) {
                break;
            }

            word_start = word.start - cursor;
            word_end = word_start + word.length;
        }

        for (char* ptr = cursor + word_start; ptr < cursor + word_end; ptr++) {
            char* start = ptr;
            char* end = ptr;

            struct Boundary token_boundary = {.start = start, .end = end};

            token_boundaries[i] = token_boundary;
            i += 1;
        }
        cursor += word_end;
    }

    if (pattern != NULL) {
        regfree(&regex);
    }
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
        struct HashMap* stats = hashmap_new(token_n, sizeof(struct Token),
                                            token_hash, token_compare);

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

            struct Token* found =
                hashmap_get(stats, &(struct Token){.key = pair});
            if (found != NULL) {
                found->value++;
            } else {
                hashmap_set(stats,
                            &(struct Token){.key = strdup(pair), .value = 1});
            }

            struct Token* current_pair =
                hashmap_get(stats, &(struct Token){.key = pair});

            if (current_pair && current_pair->value > most_common_pair.value) {
                if (most_common_pair.key != NULL) {
                    free(most_common_pair.key);
                }
                most_common_pair.value = current_pair->value;
                most_common_pair.key = strdup(current_pair->key);
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

        visualize_bpe_train(most_common_pair, token_n);

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
    struct HashMap* vocab = hashmap_new(vocab_size, sizeof(struct Token),
                                        token_hash, token_compare);

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
