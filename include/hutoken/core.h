#ifndef HUTOKEN_CORE_H
#define HUTOKEN_CORE_H

#include "Python.h"
#include "fomalib.h"

#include "hutoken/hashmap.h"

void encode(char* text,
            struct HashMap* vocab,
            char* pattern,
            int tokens[],
            int* tokens_size,
            const char** special_chars,
            const char* prefix,
            bool is_byte_encoder);
PyObject* decode(PyObject* tokens,
                 char** vocab_decode,
                 int vocab_size,
                 const char** special_chars,
                 const char* prefix,
                 bool is_byte_encoder);
PyObject* initialize_foma(void);
PyObject* look_up_word(struct apply_handle* handle,
                       char* word,
                       bool only_longest);

#endif
