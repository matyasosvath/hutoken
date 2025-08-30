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
};

struct ParserState parser_init(const char* text);
bool parser_next_token(struct ParserState* state, struct TokenSlice* token);

#endif
