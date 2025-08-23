#ifndef HUTOKEN_CORE_H
#define HUTOKEN_CORE_H

#include "Python.h"
#include "fomalib.h"

#include "hutoken/hashmap.h"
#include "hutoken/lib.h"

void encode(struct EncodeTask* task);
void decode(struct DecodeTask* task);
PyObject* initialize_foma(void);
PyObject* look_up_word(struct apply_handle* handle,
                       char* word,
                       bool only_longest);

#endif
