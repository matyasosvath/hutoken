#ifndef HUTOKEN_PRETOKENIZER_H
#define HUTOKEN_PRETOKENIZER_H

#include <stdbool.h>

int utf8_char_length(const unsigned char* c);

char* pretokenizer_encode(const char* text,
                          const char** special_chars,
                          const char* prefix,
                          bool is_byte_encoder);
char* pretokenizer_decode(const char* text,
                          const char** special_chars,
                          const char* prefix);

#endif
