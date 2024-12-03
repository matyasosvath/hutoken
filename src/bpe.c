#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <regex.h>

#include "helper.c"
#include "hashmap.c"


void create_words(
    char *text,
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
    char *cursor = text;
    int i = 0;

    while (regexec(&regex, cursor, 1, &match, 0) == 0)
    {
        int word_start = match.rm_so;
        int word_end = match.rm_eo;

        for (char *ptr = cursor + word_start; ptr < cursor + word_end; ptr++) {

            char *start = ptr;
            char *end = ptr;

            Boundary token_boundary = {start, end};

            token_boundaries[i] = token_boundary;
            i += 1;
        }
        cursor += word_end;
    }

    regfree(&regex);
}


void visualize_bpe_train(
    char* text,
    Boundary token_boundaries[],
    struct Token current_token,
    int value,
    int token_num
)
{
    if (VISUALIZE)
    {
        printf("Most common pair: '%s', rank: %d\n", current_token.key, current_token.value);
        printf("New token '%s', value: %d\n\n", current_token.key, value);
    }
}


void bpe_train_core(
    struct HashMap *vocab,
    char *text,
    Boundary token_boundaries[],
    int token_num,
    int vocab_size
)
{
    int token_n = token_num;
    struct Token prev_common_pair = {"", -1};
    struct Token most_common_pair = {"", -1};

    while (vocab->count < vocab_size)
    {
        struct HashMap *stats = hashmap_new(token_n);

        // find most common pair -> next token

        for (int i = 0; i < token_num -1; i++)
        {
            char *s1 = token_boundaries[i].start;
            char *e1 = token_boundaries[i].end;
            int l1 = (e1 - s1) + 1;

            char *s2 = token_boundaries[i+1].start;
            char *e2 = token_boundaries[i+1].end;
            int l2 = (e2 - s2) + 1;

            int len = l1 + l2;
            char pair[len + 1];

            strncpy(pair, s1, l1);
            strncpy(pair + l1, s2, l2);
            pair[len] = '\0';

            int freq = hashmap_get(stats, &(struct Token){.key = pair});
            if (freq)
            {
                hashmap_set(stats, &(struct Token){.key = strdup(pair), .value = ++freq});
            }
            else
            {
                int initial_freq = 1;
                hashmap_set(stats, &(struct Token){.key = strdup(pair), .value = initial_freq});
            }

            int rank = hashmap_get(stats, &(struct Token){.key = pair});

            if (most_common_pair.value < rank)
            {
                most_common_pair.value = rank;
                most_common_pair.key = strdup(pair);
            }
        }

        int token = vocab->count + 1;

        // add new token

        hashmap_set(vocab, &(struct Token){.key=most_common_pair.key, .value=token});

        // merge that most common pair in all tokens, i.e.
        // update token boundaries for the new token everywhere

        int j = 0;
        Boundary new_token_boundaries[token_n];

        for (int i = 0; i < token_n - 1; i++)
        {
            char *s1 = token_boundaries[i].start;
            char *e1 = token_boundaries[i].end;
            int l1 = (e1 - s1) + 1;

            char *s2 = token_boundaries[i+1].start;
            char *e2 = token_boundaries[i+1].end;
            int l2 = (e2 - s2) + 1;

            int len = l1 + l2;
            char pair[len + 1];

            strncpy(pair, s1, l1);
            strncpy(pair + l1, s2, l2);
            pair[len] = '\0';

            if (strcmp(pair, most_common_pair.key) == 0)
            {
                Boundary new_token_boundary = {s1, e2};
                new_token_boundaries[j] = new_token_boundary;
                j++;
                i++;
            }
            else {
                new_token_boundaries[j] = token_boundaries[i];
                j++;
            }
        }

        for (int k = 0; k < j; k++) {
            token_boundaries[k] = new_token_boundaries[k];
        }
        token_n = j;

        visualize_bpe_train(text, token_boundaries, most_common_pair, token, token_n);

        hashmap_free(stats);

        if (strcmp(prev_common_pair.key, most_common_pair.key) == 0) {
            break;
        } else {
            prev_common_pair.key = most_common_pair.key;
            prev_common_pair.value = most_common_pair.value;
        }

        most_common_pair.value = -1;
        most_common_pair.key = "\0";
    }
}


void bpe_train(char *text, const int vocab_size, const char *pattern)
{

    char *k;
    struct HashMap *vocab = hashmap_new(vocab_size);

    // add tokens for each individual byte value

    for (int i = 0; i < 256; i++)
    {

        char key[2];
        key[0] = (char)i; // store ascii character
        key[1] = '\0';

        k = strdup(key);
        hashmap_set(vocab, &(struct Token){.key = k, .value = i});
    }

    int token_num = strlen(text);
    Boundary token_boundaries[token_num];

    create_words(text, pattern, token_boundaries, token_num);

    bpe_train_core(vocab, text, token_boundaries, token_num, vocab_size);

    FILE *file = fopen("./vocabs/vocab.txt", "w");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file for writing.\n");
        return;
    }

    size_t iter = 0;
    void *item;
    while (hashmap_iter(vocab, &iter, &item))
    {
        const struct Token *token = item;
        if (!strlen(token->key)) { // handle null character explicitly
            fprintf(file, "0x00");
        }
        for (const char *ptr = token->key; *ptr != '\0'; ptr++) {
            fprintf(file, "0x%02X", (unsigned char)*ptr);
        }
        fprintf(file, " == %d\n", token->value);
    }

    fclose(file);
    hashmap_free(vocab);
    free(k);
}