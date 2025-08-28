#ifndef HUTOKEN_BPE_H
#define HUTOKEN_BPE_H

#include <stdint.h>

struct Token {
    char* key;
    int value;
};

struct MergeRule {
    int rank;
    int left_id;
    int right_id;
    int merge_id;
};

uint64_t token_hash(const void* item);
int token_compare(const void* a, const void* b);

void bpe_train(char* text,
               const int vocab_size,
               const char* pattern,
               char* vocab_file_name);

#endif
