#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <regex.h>

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

        // printf("before merge\n");
        // for (int i = 0; i < *token_num; i++)
        // {
        //     printf("token[%d]: '", i);
        //     char *s = token_boundaries[i].start;
        //     char *e = token_boundaries[i].end;
        //     for (char *ptr = s; ptr <= e; ptr++) {
        //         putchar(*ptr);
        //     }
        //     printf("'\n");
        // }

        // merge pairs, leave rest unchanged
        token_boundaries[min_idx].end = token_boundaries[min_idx + 1].end;

        for (int i = min_idx + 1; i < *token_num - 1; i++)
        {
            token_boundaries[i].start = token_boundaries[i + 1].start;
            token_boundaries[i].end = token_boundaries[i + 1].end;
        }
        *token_num -= 1;

        // printf("after merge\n");
        // for (int i = 0; i < *token_num; i++)
        // {
        //     printf("token[%d]: '", i);
        //     char *s = token_boundaries[i].start;
        //     char *e = token_boundaries[i].end;
        //     for (char *ptr = s; ptr <= e; ptr++) {
        //         putchar(*ptr);
        //     }
        //     printf("'\n");
        // }
        // printf("\n\n");
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

void encode(char *text, struct HashMap *vocab, char *pattern, int tokens[], int *tokens_size)
{

    regex_t regex;

    int r = regcomp(&regex, pattern, REG_EXTENDED);
    if (r)
    {
        PyErr_SetString(PyExc_RuntimeError, "Regex could not be compiled.");
    }

    regmatch_t match;

    char *cursor = text;

    while (regexec(&regex, cursor, 1, &match, 0) == 0)
    {

        int word_start = match.rm_so;
        int word_end = match.rm_eo;

        int word_len = word_end - word_start;

        int i = 0;
        Boundary word_token_boundaries[word_len];

        for (char *ptr = cursor + word_start; ptr < cursor + word_end; ptr++) {
            char *start = ptr;
            char *end = ptr;
            Boundary word_token_boundary = {start, end};
            word_token_boundaries[i] = word_token_boundary;
            i += 1;
        }

        // printf("Word: '");
        // for (int i = 0; i < word_len; i++)
        // {
        //     char *s = word_token_boundaries[i].start;
        //     char *e = word_token_boundaries[i].end;
        //     for (char *ptr = s; ptr <= e; ptr++) {
        //         printf("%c", *ptr);
        //     }
        // }
        // printf("'\n");

        int word_token_num = word_len;
        int word_tokens[word_len];

        bpe_encode(vocab, word_token_boundaries, word_tokens, &word_token_num);

        // printf("word token: ");
        for (int i = 0; i < word_token_num; i++)
        {
            tokens[i + *tokens_size] = word_tokens[i];
            // printf("%d ", word_tokens[i]);
        }
        // printf("\n");

        cursor += word_end;

        *tokens_size += word_token_num;
    }

    regfree(&regex);
}

PyObject *decode(PyObject *tokens, char **vocab_decode, int vocab_size)
{

    Py_ssize_t token_num = PyList_Size(tokens);

    size_t text_size = sizeof(char) * ((int)token_num + 1);

    char *text = (char *)malloc(text_size);

    if (!text)
        return PyErr_NoMemory();

    text[0] = '\0';

    for (Py_ssize_t i = 0; i < token_num; i++)
    {

        PyObject *token = PyList_GetItem(tokens, i);

        if (!PyLong_Check(token))
        {
            PyErr_SetString(PyExc_TypeError, "All elements of the list must be integers");
            free(text);
            return NULL;
        }

        int item = (int)PyLong_AsLong(token);

        if (item < 0 || item >= vocab_size)
        {
            PyErr_SetString(PyExc_ValueError, "Element must be non-negative and less then vocab size.");
            free(text);
            return NULL;
        }

        const char *word = vocab_decode[item];
        size_t word_len = strlen(word);

        size_t current_len = strlen(text);

        if (current_len + word_len + 1 >= text_size)
        {

            int buffer_size = current_len + TEXT_SIZE_INCREMENT + word_len + 1;

            char *new_text = realloc(text, buffer_size);

            if (new_text == NULL)
            {
                free(text);
                return PyErr_NoMemory();
            }
            text = new_text;
        }

        strcat(text, word);
    }

    PyObject *result = PyUnicode_FromString(text);

    free(text);

    return result;
}
