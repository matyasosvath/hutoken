#include "hutoken/pretokenizer.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/ac.h"
#include "hutoken/arena.h"
#include "hutoken/helper.h"
#include "hutoken/string.h"
#include "hutoken/taskqueue.h"

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
    log_debug("Starting pretokenizer_encode with text: %s and prefix: %s", text,
              prefix ? prefix : "NULL");

    struct String result_str;
    size_t text_len = strlen(text);
    size_t initial_capacity = text_len * 1.5 + (prefix ? strlen(prefix) : 0);
    if (string_with_capacity(&result_str, initial_capacity) != STRING_SUCCESS) {
        return NULL;
    }

    if (prefix) {
        if (string_append(&result_str, prefix) != STRING_SUCCESS) {
            string_release(&result_str);
            return NULL;
        }
    }

    const unsigned char* p = (const unsigned char*)text;
    while (*p != '\0') {
        const char* replacement = special_chars[*p];
        int char_len = is_byte_encoder ? 1 : utf8_char_length(p);

        if (replacement != NULL) {
            if (string_append(&result_str, replacement) != STRING_SUCCESS) {
                string_release(&result_str);
                return NULL;
            }
        } else {
            if (is_byte_encoder && *p >= 0x80) {
                char byte_pair[2];
                byte_pair[0] = 0xC0 | (*p >> 6);
                byte_pair[1] = 0x80 | (*p & 0x3F);
                if (string_append_n(&result_str, byte_pair, 2) !=
                    STRING_SUCCESS) {
                    string_release(&result_str);
                    return NULL;
                }
            } else {
                if (string_append_n(&result_str, (const char*)p, char_len) !=
                    STRING_SUCCESS) {
                    string_release(&result_str);
                    return NULL;
                }
            }
        }
        p += char_len;
    }

    const char* final_c_str = string_c_str(&result_str);
    size_t final_len = string_len(&result_str);
    char* final_result = (char*)malloc(final_len + 1);

    if (!final_result) {
        string_release(&result_str);
        return NULL;
    }

    memcpy(final_result, final_c_str, final_len + 1);

    string_release(&result_str);

    log_debug("Finished pretokenizer_encode: %s", final_result);
    return final_result;
}

char* pretokenizer_encode_arena(struct Arena* arena,
                                const char* text,
                                const char** special_chars,
                                const char* prefix,
                                bool is_byte_encoder) {
    if (!text) {
        return NULL;
    }
    log_debug("Starting pretokenizer_encode with text: %s and prefix: %s", text,
              prefix ? prefix : "NULL");

    struct String result_str;
    size_t text_len = strlen(text);
    size_t initial_capacity = text_len * 1.5 + (prefix ? strlen(prefix) : 0);
    if (string_with_capacity_arena(&result_str, arena, initial_capacity) !=
        STRING_SUCCESS) {
        return NULL;
    }

    if (prefix) {
        if (string_append_arena(&result_str, arena, prefix) != STRING_SUCCESS) {
            return NULL;
        }
    }

    const unsigned char* p = (const unsigned char*)text;
    while (*p != '\0') {
        const char* replacement = special_chars[*p];
        int char_len = is_byte_encoder ? 1 : utf8_char_length(p);

        if (replacement != NULL) {
            if (string_append_arena(&result_str, arena, replacement) !=
                STRING_SUCCESS) {
                return NULL;
            }
        } else {
            if (is_byte_encoder && *p >= 0x80) {
                char byte_pair[2];
                byte_pair[0] = 0xC0 | (*p >> 6);
                byte_pair[1] = 0x80 | (*p & 0x3F);
                if (string_append_n_arena(&result_str, arena, byte_pair, 2) !=
                    STRING_SUCCESS) {
                    return NULL;
                }
            } else {
                if (string_append_n_arena(&result_str, arena, (const char*)p,
                                          char_len) != STRING_SUCCESS) {
                    return NULL;
                }
            }
        }
        p += char_len;
    }

    const char* final_c_str = string_c_str(&result_str);
    size_t final_len = string_len(&result_str);
    char* final_result = (char*)arena_alloc(arena, final_len + 1);

    if (!final_result) {
        return NULL;
    }

    memcpy(final_result, final_c_str, final_len + 1);

    log_debug("Finished pretokenizer_encode: %s", final_result);
    return final_result;
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

char* pretokenizer_decode(const char* text, const struct DecodeContext* ctx) {
    if (!text) {
        return NULL;
    }
    log_debug(
        "Starting pretokenize_decode function with text: %s and prefix: %s and "
        "byte_encode: %d",
        text, ctx->prefix, ctx->is_byte_encoder);

    size_t text_len = strlen(text);
    if (ctx->prefix) {
        size_t prefix_len = strlen(ctx->prefix);
        if (strncmp(text, ctx->prefix, prefix_len) == 0) {
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
    const char* end_of_text = text + text_len;
    const struct ACAutomaton* automaton = ctx->ac;

    if (ctx->is_byte_encoder) {
        while (p < end_of_text) {
            const struct ACNode* longest_match_node = NULL;
            size_t longest_match_len = 0;

            struct ACNode* current_node = automaton->root;
            for (const char* q = p; q < end_of_text; ++q) {
                unsigned char index = (unsigned char)*q;
                if (current_node->children[index]) {
                    current_node = current_node->children[index];
                    if (current_node->output_value != -1) {
                        longest_match_node = current_node;
                        longest_match_len = current_node->pattern_len;
                    }
                } else {
                    break;
                }
            }

            if (longest_match_node) {
                *dest++ = (unsigned char)longest_match_node->output_value;
                p += longest_match_len;
            } else {
                int bytes_in_char = 0;
                uint32_t codepoint =
                    utf8_to_codepoint((const unsigned char*)p, &bytes_in_char);
                if (codepoint < 256) {
                    *dest++ = (unsigned char)codepoint;
                } else {
                    *dest++ = '?';
                }
                p += (bytes_in_char > 0) ? bytes_in_char : 1;
            }
        }
    } else {
        while (p < end_of_text) {
            const struct ACNode* longest_match_node = NULL;
            size_t longest_match_len = 0;

            struct ACNode* current_node = automaton->root;
            for (const char* q = p; q < end_of_text; ++q) {
                unsigned char index = (unsigned char)*q;
                if (current_node->children[index]) {
                    current_node = current_node->children[index];
                    if (current_node->output_value != -1) {
                        longest_match_node = current_node;
                        longest_match_len = current_node->pattern_len;
                    }
                } else {
                    break;
                }
            }

            if (longest_match_node) {
                *dest++ = (unsigned char)longest_match_node->output_value;
                p += longest_match_len;
                log_debug("Matched special token of length %zu",
                          longest_match_len);
            } else {
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
    log_debug("Finished pretokenize_decode function: %s", result);

    return result;
}
