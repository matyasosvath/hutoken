#ifndef HUTOKEN_BPE_H
#define HUTOKEN_BPE_H

struct Token {
    char* key;
    int value;
};

void bpe_train(char* text,
               const int vocab_size,
               const char* pattern,
               char* vocab_file_name);

#endif
