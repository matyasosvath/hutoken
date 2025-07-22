#ifndef HUTOKEN_PRETOKENIZER_H
#define HUTOKEN_PRETOKENIZER_H

char* pretokenizer_encode(const char* text,
                          const char** special_chars,
                          const char* prefix);
char* pretokenizer_decode(const char* text,
                          const char** special_chars,
                          const char* prefix);

#endif
