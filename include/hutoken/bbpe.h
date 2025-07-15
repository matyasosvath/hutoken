#ifndef HUTOKEN_BBPE_H
#define HUTOKEN_BBPE_H

struct tokenPair{
    int id1;
    int id2;
    int freq;
};

void bbpe_train(char* text,
               const int vocab_size,
               char* vocab_file_name);


#endif
