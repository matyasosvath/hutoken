#include "hutoken/parser.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "unicode_categories.inc"

static const char* consume_while(const char* p,
                                 const char* end,
                                 bool (*predicate)(uint32_t));
static bool is_custom_alpha(uint32_t cp);
static bool is_digit(uint32_t cp);
static bool is_other(uint32_t cp);
static bool is_whitespace(uint32_t cp);
static uint32_t decode_utf8(const char** s, const char* end);
static size_t contraction_length(const char* p, const char* end);
static bool in_unicode_ranges(uint32_t cp,
                              const struct UnicodeRange* ranges,
                              size_t range_count);

struct ParserState parser_init(const char* text) {
    if (!text) {
        text = "";
    }

    return parser_init_n(text, strlen(text));
}

struct ParserState parser_init_n(const char* text, size_t length) {
    if (!text) {
        text = "";
        length = 0;
    }
    return (struct ParserState){.current_pos = text, .end = text + length};
}

bool parser_next_token(struct ParserState* state, struct TokenSlice* token) {
    if (!state || !state->current_pos || state->current_pos >= state->end) {
        return false;
    }

    const char* p = state->current_pos;
    token->start = p;
    const char* end = NULL;

    size_t contraction_len = contraction_length(p, state->end);
    if (contraction_len > 0) {
        end = p + contraction_len;
        token->length = contraction_len;
        state->current_pos = end;
        return true;
    }

    const char* s = p;
    if (s < state->end && *s == ' ') {
        s++;
    }
    const char* s_after_space = s;
    s = consume_while(s, state->end, is_custom_alpha);
    if (s > s_after_space) {
        end = s;
        token->length = end - token->start;
        state->current_pos = end;
        return true;
    }

    s = p;
    if (s < state->end && *s == ' ') {
        s++;
    }
    s_after_space = s;
    s = consume_while(s, state->end, is_digit);
    if (s > s_after_space) {
        end = s;
        token->length = end - token->start;
        state->current_pos = end;
        return true;
    }

    s = p;
    if (s < state->end && *s == ' ') {
        s++;
    }
    s_after_space = s;
    s = consume_while(s, state->end, is_other);
    if (s > s_after_space) {
        end = s;
        token->length = end - token->start;
        state->current_pos = end;
        return true;
    }

    s = consume_while(p, state->end, is_whitespace);
    if (s > p) {
        // GPT-2's whitespace branch leaves the final whitespace character
        // before a non-whitespace token. An ordinary space is then attached
        // to that token; other whitespace remains its own token. For example,
        // "    word" splits as "   " and " word", while "\n\nword"
        // splits as "\n", "\n", and "word".
        const char* first_end = p;
        (void)decode_utf8(&first_end, state->end);
        if (s < state->end && s > first_end) {
            do {
                s--;
            } while (s > p && ((unsigned char)*s & 0xC0) == 0x80);
        }
        end = s;
        token->length = end - token->start;
        state->current_pos = end;
        return true;
    }

    end = p + 1;
    token->length = end - token->start;
    state->current_pos = end;
    return true;
}

static size_t contraction_length(const char* p, const char* end) {
    if (p >= end || *p != '\'' || end - p < 2) {
        return 0;
    }

    if (p[1] == 's' || p[1] == 't' || p[1] == 'm' || p[1] == 'd') {
        return 2;
    }
    if (end - p >= 3 &&
        ((p[1] == 'r' && p[2] == 'e') || (p[1] == 'v' && p[2] == 'e') ||
         (p[1] == 'l' && p[2] == 'l'))) {
        return 3;
    }
    return 0;
}

static const char* consume_while(const char* p,
                                 const char* end,
                                 bool (*predicate)(uint32_t)) {
    while (p < end) {
        const char* next_p = p;
        uint32_t cp = decode_utf8(&next_p, end);
        if (!predicate(cp)) {
            break;
        }
        p = next_p;
    }
    return p;
}

static bool is_custom_alpha(uint32_t cp) {
    return in_unicode_ranges(cp, UNICODE_LETTER_RANGES,
                             UNICODE_LETTER_RANGE_COUNT);
}

static bool is_digit(uint32_t cp) {
    return in_unicode_ranges(cp, UNICODE_NUMBER_RANGES,
                             UNICODE_NUMBER_RANGE_COUNT);
}

static bool in_unicode_ranges(uint32_t cp,
                              const struct UnicodeRange* ranges,
                              size_t range_count) {
    size_t low = 0;
    size_t high = range_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (cp < ranges[middle].first) {
            high = middle;
        } else if (cp > ranges[middle].last) {
            low = middle + 1;
        } else {
            return true;
        }
    }
    return false;
}

static bool is_other(uint32_t cp) {
    return !is_whitespace(cp) && !is_custom_alpha(cp) && !is_digit(cp);
}

static bool is_whitespace(uint32_t cp) {
    return (cp >= 0x0009 && cp <= 0x000D) || cp == 0x0020 || cp == 0x0085 ||
           cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

static uint32_t decode_utf8(const char** s, const char* end) {
    const unsigned char* p = (const unsigned char*)*s;
    size_t remaining = (size_t)(end - *s);
    if (remaining == 0) {
        return UINT32_MAX;
    }

    uint32_t cp = 0;
    int len = 0;

    if (*p < 0x80) {
        cp = p[0];
        len = 1;
    } else if ((*p & 0xE0) == 0xC0 && remaining >= 2) {
        if (*p < 0xC2 || (p[1] & 0xC0) != 0x80) {
            goto invalid;
        }
        cp = ((uint32_t)(p[0] & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
        len = 2;
    } else if ((*p & 0xF0) == 0xE0 && remaining >= 3) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) {
            goto invalid;
        }
        if ((*p == 0xE0 && p[1] < 0xA0) || (*p == 0xED && p[1] >= 0xA0)) {
            goto invalid;
        }
        cp = ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) |
             (uint32_t)(p[2] & 0x3F);
        len = 3;
    } else if ((*p & 0xF8) == 0xF0 && remaining >= 4) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 ||
            (p[3] & 0xC0) != 0x80) {
            goto invalid;
        }
        if (*p > 0xF4 || (*p == 0xF0 && p[1] < 0x90) ||
            (*p == 0xF4 && p[1] >= 0x90)) {
            goto invalid;
        }
        cp = ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
             ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
        len = 4;
    } else {
        goto invalid;
    }

    *s += len;
    return cp;

invalid:
    (*s)++;
    return UINT32_MAX;
}
