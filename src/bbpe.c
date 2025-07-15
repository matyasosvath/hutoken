#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/bbpe.h"
#include "hutoken/bpe.h"
#include "hutoken/hashmap.h"
#include "hutoken/helper.h"

int find_pair(struct tokenPair* pairs, size_t pair_count, int id1, int id2) {
    for (size_t i = 0; i < pair_count; i++) {
        if (pairs[i].id1 == id1 && pairs[i].id2 == id2) {
            return (int)i;
        }
    }
    return -1;
}

void find_most_common_pair(size_t token_n,
                           const int* token_ids,
                           struct tokenPair* most_common_pair) {
    struct tokenPair* pairs = malloc((token_n - 1) * sizeof(struct tokenPair));
    size_t pair_count = 0;

    for (size_t i = 0; i < token_n - 1; i++) {
        int id1 = token_ids[i];
        int id2 = token_ids[i + 1];

        int index = find_pair(pairs, pair_count, id1, id2);
        if (index >= 0) {
            pairs[index].freq++;
        } else {
            pairs[pair_count++] =
                (struct tokenPair){.id1 = id1, .id2 = id2, .freq = 1};
        }

        if (pairs[index].freq > most_common_pair->freq) {
            most_common_pair->freq = pairs[index].freq;
            most_common_pair->id1 = id1;
            most_common_pair->id2 = id2;
        }
    }

    free(pairs);
}

size_t merge_pair_in_token_ids(int* token_ids,
                               size_t token_n,
                               int id1,
                               int id2,
                               int new_id) {
    size_t write = 0;

    for (size_t read = 0; read < token_n;) {
        if (read < token_n - 1 && token_ids[read] == id1 &&
            token_ids[read + 1] == id2) {
            token_ids[write++] = new_id;
            read += 2;
        } else {
            token_ids[write++] = token_ids[read++];
        }
    }

    return write;
}

void bbpe_train_core(struct HashMap* vocab,
                     int* token_ids,
                     size_t tokens_length,
                     size_t vocab_size) {
    size_t token_n = tokens_length;
    struct tokenPair most_common_pair = {-1, -1, -1};
    struct tokenPair prev_common_pair = {-1, -1, -1};

    while (vocab->count < vocab_size) {
        find_most_common_pair(token_n, token_ids, &most_common_pair);
        printf("Most common pair: (%d, %d) - %d times\n", most_common_pair.id1,
               most_common_pair.id2, most_common_pair.freq);
        if (most_common_pair.freq <= 1) {
            printf("No more pairs found, stopping training.\n");
            break;
        }

        int token = (int)(vocab->count);
        char* s1 = hashmap_get_key(vocab, most_common_pair.id1);
        char* s2 = hashmap_get_key(vocab, most_common_pair.id2);
        char* pair = (char*)malloc(strlen(s1) + strlen(s2) + 1);
        if (!pair) {
            char error_msg[256];
            printf(error_msg, sizeof(error_msg),
                   "Couldn't malloc memory for pair");
            PyErr_SetString(PyExc_MemoryError, error_msg);
            return;
        }

        size_t l1 = strlen(s1);
        size_t l2 = strlen(s2);
        memcpy(pair, s1, l1);
        memcpy(pair + l1, s2, l2);
        pair[l1 + l2] = '\0';

        printf("New token: %s with value %d\n", pair, token);
        hashmap_set(vocab, &(struct Token){.key = pair, .value = token});

        token_n =
            merge_pair_in_token_ids(token_ids, token_n, most_common_pair.id1,
                                    most_common_pair.id2, token);
        printf("Merged pairs, new token count: %zu\n", token_n);

        if (prev_common_pair.id1 == most_common_pair.id1 &&
            prev_common_pair.id2 == most_common_pair.id2) {
            printf("No change in most common pair, stopping training.\n");
            free(pair);
            break;
        }

        prev_common_pair.id1 = most_common_pair.id1;
        prev_common_pair.id2 = most_common_pair.id2;
        prev_common_pair.freq = most_common_pair.freq;
        most_common_pair.id1 = -1;
        most_common_pair.id2 = -1;
        most_common_pair.freq = -1;
    }
}

void bbpe_train(char* text, const int vocab_size, char* vocab_file_name) {
    char* k = NULL;
    struct HashMap* vocab = hashmap_new(vocab_size);

    for (int i = 0; i < 256; i++) {
        char key[2];
        key[0] = (char)i;
        key[1] = '\0';

        k = strdup(key);
        hashmap_set(vocab, &(struct Token){.key = k, .value = i});
    }

    size_t token_byte_num = strlen(text);
    int* initial_token_ids = malloc(sizeof(int) * token_byte_num);
    if (!initial_token_ids) {
        (void)fputs("Couldn't malloc memory for token_ids", stderr);
        PyErr_SetString(PyExc_MemoryError,
                        "Couldn't malloc memory for token_ids");
        hashmap_free(vocab);
        free(k);
        return;
    }

    for (size_t i = 0; i < token_byte_num; i++) {
        initial_token_ids[i] = (unsigned char)text[i];
    }

    bbpe_train_core(vocab, initial_token_ids, token_byte_num, vocab_size);

    printf("Saving vocabulary");
    save_vocab(vocab, vocab_file_name);

    hashmap_free(vocab);
    free(k);
    free(initial_token_ids);
}
