#ifndef HUTOKEN_PRETOKENIZER_H
#define HUTOKEN_PRETOKENIZER_H

int utf8_char_length(char c);

char* pretokenizer_encode(const char* text,
                          const char** special_chars,
                          const char* prefix,
                          char** is_special_out);
char* pretokenizer_decode(const char* text,
                          const char** special_chars,
                          const char* prefix);

#endif
