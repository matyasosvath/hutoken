#ifndef HELPER
#define HELPER

#include <Python.h>

#ifndef PyExc_BufferError
#define PyExc_BufferError PyExc_RuntimeError
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/stat.h>

#include "hashmap.c"

#define VISUALIZE 1
#define TEXT_SIZE_INCREMENT 50


typedef struct {
    char *start;
    char *end;
} Boundary;


void visualize(int arr[], char *text, int n)
{

    if (VISUALIZE)
    {
        printf("Processing:\n");
        for (int i = 0; i < n; i++)
        {
            printf("%c ", text[i]);
        }
        printf("\n");
        for (int i = 0; i < n; i++)
        {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
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

void hex_str_to_ascii(const char *hex_str, char *ascii_str, size_t ascii_str_size) {
    log_debug("Starting hex_str_to_ascii with input: %s", hex_str);

    size_t i = 0;
    while (*hex_str != '\0') {
        if (*hex_str == '0' && *(hex_str + 1) == 'x') {
            hex_str += 2;

            if (hex_str[0] != '\0' && hex_str[1] != '\0') {
                char hexValue[3] = { hex_str[0], hex_str[1], '\0' };
                int charValue = (int)strtol(hexValue, NULL, 16);

                log_debug("Parsed hex value: %s -> ASCII char: %c (decimal: %d)", hexValue, (char)charValue, charValue);

                if (i >= ascii_str_size - 1) {
                    log_debug("Error: Output buffer overflow in hex_str_to_ascii. Buffer size: %zu, Current index: %zu", ascii_str_size, i);
                    PyErr_SetString(PyExc_BufferError, "Output buffer overflow in hex_str_to_ascii");
                    ascii_str[0] = '\0';
                    return;
                }

                ascii_str[i++] = (char)charValue;
            } else {
                log_debug("Error: Incomplete hex pair at position: %s", hex_str);
            }

            hex_str += 2; 
        } else {
            log_debug("Skipping non-hex character: %c", *hex_str);
            hex_str++;
        }
    }

    ascii_str[i] = '\0';
    log_debug("Completed hex_str_to_ascii. Result: %s", ascii_str);
}

int save_vocab(struct HashMap *vocab, char *file_name) {

    const char *home_dir = getenv("HOME");

    if (home_dir == NULL) {
        fprintf(stderr, "Unable to get HOME environment variable.\n");
        return EXIT_FAILURE;
    }

    char dir_path[1024];
    snprintf(dir_path, sizeof(dir_path), "%s/config", home_dir);

    struct stat st = {0};
    if (stat(dir_path, &st) == -1) {
        if (mkdir(dir_path, 0700) == 0) {
            printf("Directory created: %s\n", dir_path);
        } else {
            fprintf(stderr, "Error creating directory.\n");
            fprintf(stderr, "Failed to save vocab.\n");
            return -1;
        }
    } else {
        printf("Directory already exists: %s\n", dir_path);
    }

    char file_path[1024];
    snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, file_name);

    FILE *file = fopen(file_path, "w");
    if (file == NULL) {
        perror("Error creating file.\n");
        return EXIT_FAILURE;
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

    printf("Vocab saved to: %s\n", file_path);

    return EXIT_SUCCESS;
}

#endif