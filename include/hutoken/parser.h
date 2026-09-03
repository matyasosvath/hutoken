#ifndef HUTOKEN_PARSER_H
#define HUTOKEN_PARSER_H

#include <stdbool.h>
#include <stddef.h>

struct TokenSlice {
    const char* start;
    size_t length;
};

struct ParserState {
    const char* current_pos;
    const char* end;
};

struct ParserState parser_init(const char* text);
struct ParserState parser_init_n(const char* text, size_t length);
bool parser_next_token(struct ParserState* state, struct TokenSlice* token);

#endif
