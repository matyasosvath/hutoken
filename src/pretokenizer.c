#include "hutoken/pretokenizer.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/helper.h"

int utf8_char_length(char c) {
    if ((c & 0x80) == 0x00) {
        return 1;
    }
    if ((c & 0xE0) == 0xC0) {
        return 2;
    }
    if ((c & 0xF0) == 0xE0) {
        return 3;
    }
    if ((c & 0xF8) == 0xF0) {
        return 4;
    }
    return 1;
}

char* pretokenizer_encode(const char* text,
                          const char** special_chars,
                          const char* prefix,
                          char** is_special_out) {
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

    for (const char* p = text; *p != '\0'; ++p) {
        unsigned char current_char = (unsigned char)*p;
        const char* replacement = special_chars[current_char];

        if (replacement != NULL) {
            size_t repl_len = strlen(replacement);
            memcpy(dest, replacement, repl_len);
            memset(is_special_dest, 1, repl_len);
            dest += repl_len;
            is_special_dest += repl_len;
        } else {
            *dest++ = *p;
            *is_special_dest++ = 0;
        }
    }
    *dest = '\0';
    *is_special_dest = '\0';

    if (is_special_out) {
        *is_special_out = is_special;
    } else {
        free(is_special);
    }
    return result;
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
