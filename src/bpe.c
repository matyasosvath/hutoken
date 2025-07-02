#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "helper.c"
#include "hashmap.c"


void create_words(
    char *text,
    const char *pattern,
    Boundary token_boundaries[],
    int token_num
)
{
    int error_number;
    PCRE2_SIZE error_offset;
    pcre2_code *regex = pcre2_compile(
        (PCRE2_SPTR) pattern,
        PCRE2_ZERO_TERMINATED,
        0,
        &error_number,
        &error_offset,
        NULL0);

    if(regex == NULL){
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(error_number, buffer, sizeof(buffer));
        fprintf(stderr, "PCRE2 compilation failed at offset %d: %s\n", (int)error_offset, buffer);
        PyErr_Format(PyExc_RuntimeError, "PCRE2 compilation failed at offset %d: %s", (int)error_offset, buffer);
        return;
    }

    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(regex, NULL);
    if(match_data == NULL){
        fprintf(stderr, "Failed to create PCRE2 match data.\n");
        PyErr_SetString(PyExc_RuntimeError, "Failed to create PCRE2 match data.");
        pcre2_code_free(regex);
        return;
    }

    PCRE2_SPTR subject = (PCRE2_SPTR) text;
    PCRE2_SIZE subject_length = strlen(text);
    PCRE2_SIZE start_offset = 0;
    int i = 0;

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
            }else{
                fprintf(stderr, "PCRE2 matching error: %d\n", rc);
                PyErr_Format(PyExc_RuntimeError, "PCRE2 matching error: %d", rc);
                pcre2_match_data_free(match_data);
                pcre2_code_free(regex);
                return;
            }
        }

        PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE match_start = ovector[0];
        PCRE2_SIZE match_end = ovector[1];

        for(char *ptr = text + match_start; ptr < text + match_end; ptr++){
            char *start = ptr;
            char *end = ptr;

            Boundary token_boundary = {start, end};

            token_boundaries[i] = token_boundary;
            i += 1;
        }

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


void bpe_train(char *text, const int vocab_size, const char *pattern, char *vocab_file_name)
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

    save_vocab(vocab, vocab_file_name);

    hashmap_free(vocab);
    free(k);
}