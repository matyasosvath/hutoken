#include "hutoken/parser.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>

static const char* consume_while(const char* p, bool (*predicate)(uint32_t));
static bool is_custom_alpha(uint32_t cp);
static bool is_digit(uint32_t cp);
static bool is_other(uint32_t cp);
static bool is_whitespace(uint32_t cp);
static uint32_t decode_utf8(const char** s);

struct ParserState parser_init(const char* text) {
    struct ParserState state = {.current_pos = text};

    if (!text) {
        state.current_pos = "";
    }

    return state;
}

bool parser_next_token(struct ParserState* state, struct TokenSlice* token) {
    if (!state || !state->current_pos || *state->current_pos == '\0') {
        return false;
    }

    const char* p = state->current_pos;
    token->start = p;
    const char* end = p;

    const char* s = p;
    if (*s == ' ') {
        s++;
    }
    const char* s_after_space = s;
    s = consume_while(s, is_custom_alpha);
    if (s > s_after_space) {
        end = s;
        token->length = end - token->start;
        state->current_pos = end;
        return true;
    }

    s = p;
    if (*s == ' ') {
        s++;
    }
    s_after_space = s;
    s = consume_while(s, is_digit);
    if (s > s_after_space) {
        end = s;
        token->length = end - token->start;
        state->current_pos = end;
        return true;
    }

    s = p;
    if (*s == ' ') {
        s++;
    }
    s_after_space = s;
    s = consume_while(s, is_other);
    if (s > s_after_space) {
        end = s;
        token->length = end - token->start;
        state->current_pos = end;
        return true;
    }

    s = p;
    if (*s == ' ') {
        s++;
        while (*s == ' ') {
            s++;
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

static const char* consume_while(const char* p, bool (*predicate)(uint32_t)) {
    while (*p != '\0') {
        const char* next_p = p;
        uint32_t cp = decode_utf8(&next_p);
        if (cp == 0 || !predicate(cp)) {
            break;
        }
        p = next_p;
    }
    return p;
}

static bool is_custom_alpha(uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) {
        return true;
    }
    switch (cp) {
        case 0x00E1:
        case 0x00E9:
        case 0x00ED:
        case 0x00F3:
        case 0x00FA:  // á, é, í, ó, ú
        case 0x0151:
        case 0x0171:
        case 0x00FC:
        case 0x00F6:  // ő, ű, ü, ö
        case 0x00C1:
        case 0x00C9:
        case 0x00CD:
        case 0x00D3:
        case 0x00DA:  // Á, É, Í, Ó, Ú
        case 0x0150:
        case 0x0170:
        case 0x00DC:
        case 0x00D6:  // Ő, Ű, Ü, Ö
            return true;
        default:
            return false;
    }
}

static bool is_digit(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

static bool is_other(uint32_t cp) {
    return cp != 0 && !is_whitespace(cp) && !is_custom_alpha(cp) &&
           !is_digit(cp);
}

static bool is_whitespace(uint32_t cp) {
    return cp <= 255 && isspace(cp);
}

static uint32_t decode_utf8(const char** s) {
    const unsigned char* p = (const unsigned char*)*s;
    if (*p == 0) {
        return 0;
    }

    uint32_t cp = 0;
    int len = 0;

    if (*p < 0x80) {
        cp = p[0];
        len = 1;
    } else if ((*p & 0xE0) == 0xC0) {
        if ((p[1] & 0xC0) != 0x80) {
            return 0;
        }
        cp = ((uint32_t)(p[0] & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
        len = 2;
    } else if ((*p & 0xF0) == 0xE0) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) {
            return 0;
        }
        cp = ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) |
             (uint32_t)(p[2] & 0x3F);
        len = 3;
    } else if ((*p & 0xF8) == 0xF0) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 ||
            (p[3] & 0xC0) != 0x80) {
            return 0;
        }
        cp = ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
             ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
        len = 4;
    } else {
        return 0;
    }

    *s += len;
    return cp;
}
