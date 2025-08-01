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
        free(is_special);
        return result;
    } else {
        const char* p = text;
        while (*p != '\0') {
            unsigned char current_char = (unsigned char)*p;
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
        free(is_special);
        free(result);
        return NULL;
    }

    char* bdest = byte_text;

    if (dest != result) {
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
    }
    *bdest = '\0';

    free(result);
    free(is_special);

    return byte_text;
}

/*
 * A helper function to decode a single UTF-8 character from a byte stream
 * and return its Unicode code point.
 * Needed for tokenizers which use byte encoding.
 */
uint32_t utf8_to_codepoint(const unsigned char* p, int* bytes_read) {
    uint32_t cp = 0;
    if (p[0] < 0x80) {
        cp = p[0];
        *bytes_read = 1;
    } else if ((p[0] & 0xE0) == 0xC0) {
        cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F);
        *bytes_read = 2;
    } else if ((p[0] & 0xF0) == 0xE0) {
        cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        *bytes_read = 3;
    } else if ((p[0] & 0xF8) == 0xF0) {
        cp = ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) |
             ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        *bytes_read = 4;
    } else {
        cp = 0xFFFD;
        *bytes_read = 1;
    }
    return cp;
}

char* pretokenizer_decode(const char* text,
                          const char** special_chars,
                          const char* prefix,
                          bool byte_level) {
    if (!text) {
        return NULL;
    }

    size_t text_len = strlen(text);
    if (prefix) {
        size_t prefix_len = strlen(prefix);
        if (strncmp(text, prefix, prefix_len) == 0) {
            text += prefix_len;
            text_len -= prefix_len;
        }
    }

    char* buffer = (char*)malloc(text_len + 1);
    if (!buffer) {
        return NULL;
    }
    char* dest = buffer;

    const char* p = text;

    if (byte_level) {
        const unsigned char* up = (const unsigned char*)p;
        while (*up != '\0') {
            int bytes_in_char = 0;
            uint32_t codepoint = utf8_to_codepoint(up, &bytes_in_char);

            char current_char_str[5] = {0};
            strncpy(current_char_str, (const char*)up, bytes_in_char);

            bool matched = false;
            for (int i = 0; i < 256; i++) {
                if (special_chars[i] &&
                    strcmp(current_char_str, special_chars[i]) == 0) {
                    *dest++ = (unsigned char)i;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                if (codepoint < 256) {
                    *dest++ = (unsigned char)codepoint;
                } else {
                    *dest++ = '?';
                }
            }
            up += bytes_in_char;
        }
    } else {
        while (*p != '\0') {
            bool matched = false;
            for (int i = 0; i < 256; i++) {
                if (special_chars[i]) {
                    size_t tok_len = strlen(special_chars[i]);
                    if (strncmp(p, special_chars[i], tok_len) == 0) {
                        *dest++ = (unsigned char)i;
                        p += tok_len;
                        matched = true;
                        break;
                    }
                }
            }

            if (!matched) {
                int char_len = utf8_char_length((const unsigned char*)p);
                memcpy(dest, p, char_len);
                dest += char_len;
                p += char_len;
            }
        }
    }

    *dest = '\0';

    size_t final_len = dest - buffer;
    char* result = (char*)malloc(final_len + 1);
    if (!result) {
        free(buffer);
        return NULL;
    }
    memcpy(result, buffer, final_len);
    result[final_len] = '\0';
    free(buffer);

    return result;
}
