#ifndef HUTOKEN_CORE_H
#define HUTOKEN_CORE_H

#include "Python.h"

#include "hashmap.h"

void encode(char *text, struct HashMap *vocab, char *pattern, int tokens[], int *tokens_size);
PyObject *decode(PyObject *tokens, char **vocab_decode, int vocab_size);

#endif
