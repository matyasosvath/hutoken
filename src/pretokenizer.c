#include "hutoken/pretokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/helper.h"

char* pretokenizer_encode(const char* text, const char** special_chars) {
    if (!text) {
        return NULL;
    }

    size_t new_len = 0;
    for (const char* p = text; *p != '\0'; p++) {
        unsigned char current_char = (unsigned char)*p;

        if (special_chars[current_char] != NULL) {
            new_len += strlen(special_chars[current_char]);
        } else {
            new_len += 1;
        }
    }

    char* result = (char*)malloc(new_len + 1);
    if (!result) {
        return NULL;
    }

    char* dest = result;
    for (const char* p = text; *p != '\0'; ++p) {
        unsigned char current_char = (unsigned char)*p;
        const char* replacement = special_chars[current_char];

        if (replacement != NULL) {
            size_t repl_len = strlen(replacement);
            memcpy(dest, replacement, repl_len);
            dest += repl_len;
        } else {
            *dest = *p;
            ++dest;
        }
    }
    *dest = '\0';

    return result;
}
