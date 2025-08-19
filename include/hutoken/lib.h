#ifndef HUTOKEN_LIB_H
#define HUTOKEN_LIB_H

#include <stdbool.h>

#include "Python.h"

struct EncodeContext {
    bool initialized_encode;
    struct HashMap* vocab_encode;
    char* pattern;
    char* special_chars[256];
    char* prefix;
    bool is_byte_encoder;
};

struct DecodeContext {
    bool initialized_decode;
    char** vocab_decode;
    int vocab_size_decode;
    char* special_chars[256];
    char* prefix;
    bool is_byte_encoder;
};

struct ThreadTask {
    char* text;
    struct EncodeContext* ctx;
    int* tokens;
    int* tokens_size;
};

PyObject* p_bpe_train(PyObject* self, PyObject* args);
PyObject* p_bbpe_train(PyObject* self, PyObject* args);
PyObject* p_encode(PyObject* self, PyObject* args, PyObject* kwargs);
PyMODINIT_FUNC PyInit__hutoken(void);
PyObject* p_initialize_foma(PyObject* self);
PyObject* p_look_up_word(PyObject* self, PyObject* args);

#endif
