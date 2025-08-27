#ifndef HUTOKEN_HELPER_H
#define HUTOKEN_HELPER_H

#include <stddef.h>

#include "hutoken/bbpe.h"
#include "hutoken/bpe.h"
#include "hutoken/hashmap.h"

#define TEXT_SIZE_INCREMENT 50

struct Boundary {
    char* start;
    char* end;
};

void log_debug(const char* format, ...);
void visualize(int arr[], char* text, int n);
void visualize_bbpe_train(struct TokenPair current_token, size_t value);
void visualize_bpe_train(struct Token current_token, size_t value);
void hex_str_to_ascii(const char* hex_str,
                      char* ascii_str,
                      size_t ascii_str_size);
int save_vocab(struct HashMap* vocab, char* file_name);
int count_char(const char* source, char target);

#endif
