#include "hutoken/helper.h"

#include "Python.h"

#ifndef PyExc_BufferError
#define PyExc_BufferError PyExc_RuntimeError
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "hutoken/hashmap.h"

#define VISUALIZE 1

void log_debug(const char* format, ...) {
    const char* debug_env = getenv("DEBUG");  // NOLINT: concurrency-mt-unsafe
    if (debug_env && strcmp(debug_env, "1") == 0) {
        time_t now = time(NULL);
        struct tm* local_time =
            localtime(&now);  // NOLINT: concurrency-mt-unsafe
        char timestamp[20];
        (void)strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S",
                       local_time);

        (void)fprintf(stderr, "[%s] DEBUG: ", timestamp);

        va_list args;
        va_start(args, format);
        (void)vfprintf(stderr, format, args);
        va_end(args);

        (void)fputs("\n", stderr);
    }
}

void visualize(int arr[], char* text, int n) {
    if (VISUALIZE) {
        (void)puts("Processing:");
        for (int i = 0; i < n; i++) {
            (void)printf("%c ", text[i]);
        }
        (void)puts("");
        for (int i = 0; i < n; i++) {
            (void)printf("%d ", arr[i]);
        }
        (void)puts("");
    }
}

void visualize_bpe_train(struct Token current_token, size_t value) {
    if (VISUALIZE) {
        (void)printf("Most common pair: '%s', rank: %d\n", current_token.key,
                     current_token.value);
        (void)printf("New token '%s', value: %ld\n\n", current_token.key,
                     value);
    }
}

void visualize_bbpe_train(struct TokenPair current_token, size_t value) {
    if (VISUALIZE) {
        (void)printf("Most common pair: (%d, %d), freq: %d\n",
                     current_token.id1, current_token.id2, current_token.freq);
        (void)printf("New token id: %zu\n\n", value);
    }
}

void hex_str_to_ascii(const char* hex_str,
                      char* ascii_str,
                      size_t ascii_str_size) {
    log_debug("Starting hex_str_to_ascii with input: %s", hex_str);

    size_t i = 0;
    while (*hex_str != '\0') {
        if (*hex_str == '0' && *(hex_str + 1) == 'x') {
            hex_str += 2;

            if (hex_str[0] != '\0' && hex_str[1] != '\0') {
                char hex_value[3] = {hex_str[0], hex_str[1], '\0'};
                unsigned int byte_value =
                    (unsigned int)strtol(hex_value, NULL, 16);

                log_debug(
                    "Parsed hex value: %s -> ASCII char: %c (decimal: %d)",
                    hex_value, (char)byte_value, byte_value);

                if (i >= ascii_str_size - 1) {
                    log_debug(
                        "Error: Output buffer overflow in hex_str_to_ascii. "
                        "Buffer size: %zu, Current index: %zu",
                        ascii_str_size, i);
                    PyErr_SetString(
                        PyExc_BufferError,
                        "Output buffer overflow in hex_str_to_ascii");
                    ascii_str[0] = '\0';
                    return;
                }

                ascii_str[i++] = (char)byte_value;
            } else {
                log_debug("Error: Incomplete hex pair at position: %s",
                          hex_str);
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

int save_vocab(struct HashMap* vocab, char* file_name) {
    const char* home_dir = getenv("HOME");  // NOLINT: concurrency-mt-unsafe

    if (home_dir == NULL) {
        (void)fputs("Unable to get HOME environment variable.", stderr);
        return EXIT_FAILURE;
    }

    char dir_path[1024];
    (void)snprintf(dir_path, sizeof(dir_path), "%s/config", home_dir);

    struct stat st = {0};
    if (stat(dir_path, &st) == -1) {
        if (mkdir(dir_path, 0700) == 0) {
            (void)printf("Directory created: %s\n", dir_path);
        } else {
            (void)fputs("Error creating directory.", stderr);
            (void)fputs("Failed to save vocab.", stderr);
            return -1;
        }
    } else {
        (void)printf("Directory already exists: %s\n", dir_path);
    }

    char file_path[1024];
    (void)snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, file_name);

    FILE* file = fopen(file_path, "w");
    if (file == NULL) {
        perror("Error creating file.\n");
        return EXIT_FAILURE;
    }

    size_t iter = 0;
    void* item = NULL;
    while (hashmap_iter(vocab, &iter, &item)) {
        const struct Token* token = item;
        if (!strlen(token->key)) {  // handle null character explicitly
            (void)fprintf(file, "0x00");
        }
        for (const char* ptr = token->key; *ptr != '\0'; ptr++) {
            (void)fprintf(file, "0x%02X", (unsigned char)*ptr);
        }
        (void)fprintf(file, " == %d\n", token->value);
    }

    (void)fclose(file);

    (void)printf("Vocab saved to: %s\n", file_path);

    return EXIT_SUCCESS;
}

int count_char(const char* source, char target) {
    int count = 0;
    while (*source) {
        if (*source == target) {
            count++;
        }
        source++;
    }
    return count;
}
