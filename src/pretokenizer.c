#include "hutoken/pretokenizer.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/helper.h"

int utf8_char_length(const unsigned char* c) {
    if ((c[0] & 0x80) == 0x00) {
        return 1;
    }
    if ((c[0] & 0xE0) == 0xC0) {
        return 2;
    }
    if ((c[0] & 0xF0) == 0xE0) {
        return 3;
    }
    if ((c[0] & 0xF8) == 0xF0) {
        return 4;
    }
    return 1;
}

char* pretokenizer_encode(const char* text,
                          const char** special_chars,
                          const char* prefix,
                          bool is_byte_encoder) {
    if (!text) {
        return NULL;
    }

    log_debug("prefix=%s", prefix);
    size_t prefix_len = (prefix != NULL) ? strlen(prefix) : 0;
    size_t new_len = prefix_len;
    for (const char* p = text; *p != '\0'; p++) {
        unsigned char current_char = (unsigned char)*p;

        if (special_chars[current_char] != NULL) {
            new_len += strlen(special_chars[current_char]);
        } else {
            new_len += 1;
        }
    }

    char* result = (char*)malloc(new_len + 1);
    char* is_special = (char*)malloc(new_len + 1);

    if (!result || !is_special) {
        free(result);
        free(is_special);
        return NULL;
    }

    char* dest = result;
    char* is_special_dest = is_special;

    if (prefix_len > 0) {
        memcpy(dest, prefix, prefix_len);
        memset(is_special_dest, 0, prefix_len);
        dest += prefix_len;
        is_special_dest += prefix_len;
    }

    if (!is_byte_encoder) {
        for (const char* p = text; *p != '\0';) {
            unsigned char current_char = (unsigned char)*p;
            int char_len = utf8_char_length((unsigned char*)p);
            const char* replacement = special_chars[current_char];

            if (replacement != NULL) {
                size_t repl_len = strlen(replacement);
                memcpy(dest, replacement, repl_len);

                dest += repl_len;
                is_special_dest += repl_len;
            } else {
                memcpy(dest, p, char_len);

                dest += char_len;
            }

            p += char_len;
        }

        *dest = '\0';
        *is_special_dest = '\0';
        return result;
    } else {
        const char* p = text;
        while (*p != '\0') {
            unsigned char current_char = (unsigned char)*p;
            int char_len = utf8_char_length(&current_char);
            const char* replacement = special_chars[current_char];

            if (replacement != NULL) {
                size_t repl_len = strlen(replacement);

                memcpy(dest, replacement, repl_len);
                memset(is_special_dest, 's', repl_len);

                dest += repl_len;
                is_special_dest += repl_len;
            } else {
                *dest++ = *p;
                *is_special_dest++ = 'n';
            }
            p++;
        }

        *dest = '\0';
        *is_special_dest = '\0';
    }
    size_t result_len = strlen(result);

    char* byte_text = malloc(result_len * 2 + 1);
    if (!byte_text) {
        free(result);
        return NULL;
    }

    char* bdest = byte_text;

    for (size_t i = 0; i < result_len; ++i) {
        unsigned char current_char = (unsigned char)result[i];

        if (is_special[i] == 'n' && current_char >= 0x80) {
            unsigned char first_byte = 0xC0 | (current_char >> 6);
            unsigned char second_byte = 0x80 | (current_char & 0x3F);
            *bdest++ = first_byte;
            *bdest++ = second_byte;
        } else {
            *bdest++ = current_char;
        }
    }
    *bdest = '\0';

    free(result);
    free(is_special);

    return byte_text;
}

char* pretokenizer_decode(const char* text,
                          const char** special_chars,
                          const char* prefix) {
    if (!text) {
        return NULL;
    }
    if (prefix) {
        size_t prefix_len = strlen(prefix);
        if (strncmp(text, prefix, prefix_len) == 0) {
            text += prefix_len;
        }
    }

    size_t max_len = strlen(text);
    char* result = (char*)malloc(max_len + 1);
    if (!result) {
        return NULL;
    }

    char* dest = result;

    for (const char* p = text; *p != '\0';) {
        bool matched = false;

        for (int i = 0; i < 256; i++) {
            const char* searched_char = special_chars[i];
            if (searched_char &&
                strncmp(p, searched_char, strlen(searched_char)) == 0) {
                *dest = (unsigned char)i;
                ++dest;
                p += strlen(searched_char);
                matched = true;
                break;
            }
        }

        if (!matched) {
            *dest = (unsigned char)*p;
            ++dest;
            ++p;
        }
    }
    *dest = '\0';

    return result;
}
