#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <regex.h>
#include <stdint.h>

#include "helper.c"
#include "hashmap.c"

void create_words(
    const uint8_t *text,
    const char *pattern,
    Boundary token_boundaries[],
    int token_num
)
{
    regex_t regex;
    int r = regcomp(&regex, pattern, REG_EXTENDED);
    if (r)
    {
        fprintf(stderr, "Regex could not be compiled.\n");
        PyErr_SetString(PyExc_RuntimeError, "Regex could not be compiled.");
        return;
    }

    regmatch_t match;
    const uint8_t *cursor = text;
    int i = 0;

    while (regexec(&regex, (const char *)cursor, 1, &match, 0) == 0)
    {
        int word_start = match.rm_so;
        int word_end = match.rm_eo;

        for (const uint8_t *ptr = cursor + word_start; ptr < cursor + word_end; ptr++) {
            const uint8_t *start = ptr;
            const uint8_t *end = ptr;
            Boundary token_boundary = {start, end};
            token_boundaries[i] = token_boundary;
            i += 1;
        }
        cursor += word_end;
    }

    regfree(&regex);
}

void bpe_train_core(
    struct HashMap *vocab,
    const uint8_t *text,
    Boundary token_boundaries[],
    int token_num,
    int vocab_size
)
{
    int token_n = token_num;
    struct {
        char *bytes;
        int len;
        int value;
    } prev_common_pair = {NULL, 0, -1}, most_common_pair = {NULL, 0, -1};

    while (vocab->count < vocab_size)
    {
        struct HashMap *stats = hashmap_new(token_n);

        // Find most common pair
        for (int i = 0; i < token_n - 1; i++)
        {
            const uint8_t *s1 = token_boundaries[i].start;
            const uint8_t *e1 = token_boundaries[i].end;
            int l1 = (int)(e1 - s1) + 1;

            const uint8_t *s2 = token_boundaries[i+1].start;
            const uint8_t *e2 = token_boundaries[i+1].end;
            int l2 = (int)(e2 - s2) + 1;

            int len = l1 + l2;
            char pair[len + 1];
            memcpy(pair, s1, l1);
            memcpy(pair + l1, s2, l2);
            pair[len] = '\0'; // Null-terminate

            struct Token probe = { .key = pair, .value = 0 };
            int freq = hashmap_get(stats, &probe);
            struct Token entry = { .key = pair, .value = freq ? freq + 1 : 1 };
            hashmap_set(stats, &entry);

            int rank = freq ? freq + 1 : 1;

            if (most_common_pair.value < rank)
            {
                if (most_common_pair.bytes) free(most_common_pair.bytes);
                most_common_pair.bytes = malloc(len + 1);
                memcpy(most_common_pair.bytes, pair, len + 1);
                most_common_pair.len = len;
                most_common_pair.value = rank;
            }
        }

        if (!most_common_pair.bytes) break; // No more pairs

        int token = vocab->count + 1;

        // Add new token to vocab
        char *vocab_key = malloc(most_common_pair.len + 1);
        memcpy(vocab_key, most_common_pair.bytes, most_common_pair.len + 1);
        struct Token vocab_entry = { .key = vocab_key, .value = token };
        hashmap_set(vocab, &vocab_entry);

        // Merge that most common pair in all tokens
        int j = 0;
        Boundary new_token_boundaries[token_n];

        for (int i = 0; i < token_n - 1; i++)
        {
            const uint8_t *s1 = token_boundaries[i].start;
            const uint8_t *e1 = token_boundaries[i].end;
            int l1 = (int)(e1 - s1) + 1;

            const uint8_t *s2 = token_boundaries[i+1].start;
            const uint8_t *e2 = token_boundaries[i+1].end;
            int l2 = (int)(e2 - s2) + 1;

            int len = l1 + l2;
            char pair[len + 1];
            memcpy(pair, s1, l1);
            memcpy(pair + l1, s2, l2);
            pair[len] = '\0';

            if (len == most_common_pair.len && memcmp(pair, most_common_pair.bytes, len + 1) == 0)
            {
                Boundary new_token_boundary = {s1, e2};
                new_token_boundaries[j] = new_token_boundary;
                j++;
                i++; // skip next
            }
            else {
                new_token_boundaries[j] = token_boundaries[i];
                j++;
            }
        }
        // If last token wasn't merged, copy it
        if (token_n > 0 && (j == 0 || new_token_boundaries[j-1].end != token_boundaries[token_n-1].end)) {
            new_token_boundaries[j] = token_boundaries[token_n-1];
            j++;
        }

        for (int k = 0; k < j; k++) {
            token_boundaries[k] = new_token_boundaries[k];
        }
        token_n = j;

        hashmap_free(stats);

        if (prev_common_pair.len == most_common_pair.len &&
            prev_common_pair.value == most_common_pair.value &&
            prev_common_pair.bytes &&
            memcmp(prev_common_pair.bytes, most_common_pair.bytes, most_common_pair.len + 1) == 0) {
            break;
        } else {
            if (prev_common_pair.bytes) free(prev_common_pair.bytes);
            prev_common_pair.bytes = malloc(most_common_pair.len + 1);
            memcpy(prev_common_pair.bytes, most_common_pair.bytes, most_common_pair.len + 1);
            prev_common_pair.len = most_common_pair.len;
            prev_common_pair.value = most_common_pair.value;
        }

        most_common_pair.value = -1;
        if (most_common_pair.bytes) { free(most_common_pair.bytes); most_common_pair.bytes = NULL; }
        most_common_pair.len = 0;
    }
    if (prev_common_pair.bytes) free(prev_common_pair.bytes);
    if (most_common_pair.bytes) free(most_common_pair.bytes);
}

void bpe_train(const uint8_t *text, const int vocab_size, const char *pattern, char *vocab_file_name)
{
    uint8_t *k;
    struct HashMap *vocab = hashmap_new(vocab_size);

    // add tokens for each individual byte value
    for (int i = 0; i < 256; i++)
    {
        char key[2] = { (char)i, '\0' };
        k = malloc(2);
        k[0] = (char)i;
        k[1] = '\0';
        struct Token entry = { .key = (char *)k, .value = i };
        hashmap_set(vocab, &entry);
    }

    int token_num = strlen((const char *)text);
    Boundary token_boundaries[token_num];

    create_words(text, pattern, token_boundaries, token_num);

    bpe_train_core(vocab, text, token_boundaries, token_num, vocab_size);

    save_vocab(vocab, vocab_file_name);

    hashmap_free(vocab);
    free(k);
}