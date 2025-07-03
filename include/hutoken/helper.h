#ifndef HUTOKEN_HELPER_H
#define HUTOKEN_HELPER_H

#include <stddef.h>

#include "hutoken/bpe.h"
#include "hutoken/hashmap.h"

#define TEXT_SIZE_INCREMENT 50

struct Boundary {
    char* start;
    char* end;
};

void log_debug(const char* format, ...);
void visualize(int arr[], char* text, int n);
void visualize_bpe_train(char* text,
                         struct Boundary token_boundaries[],
                         struct Token current_token,
                         size_t value,
                         size_t token_num);
void hex_str_to_ascii(const char* hex_str,
                      char* ascii_str,
                      size_t ascii_str_size);
int save_vocab(struct HashMap* vocab, char* file_name);

#endif
