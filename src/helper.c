#ifndef HELPER
#define HELPER

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
    size_t i = 0;  // Index for ascii_str
    while (*hex_str != '\0') {
        if (*hex_str == '0' && *(hex_str + 1) == 'x') {
            hex_str += 2;  // Skip "0x"

            if (hex_str[0] != '\0' && hex_str[1] != '\0') {
                char hexValue[3] = { hex_str[0], hex_str[1], '\0' };
                int charValue = (int)strtol(hexValue, NULL, 16);

                // Ensure there is enough space in the output buffer
                if (i >= ascii_str_size - 1) {
                    fprintf(stderr, "Error: Output buffer overflow in hex_str_to_ascii\n");
                    ascii_str[0] = '\0';  // Mark as invalid
                    return;
                }

                ascii_str[i++] = (char)charValue;
            }

            hex_str += 2;  // Move to the next potential '0x'
        } else {
            hex_str++;  // Move to the next character if '0x' not found
        }
    }
    ascii_str[i] = '\0';  // Null-terminate the resulting ASCII string
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