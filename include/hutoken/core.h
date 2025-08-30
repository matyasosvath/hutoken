#ifndef HUTOKEN_CORE_H
#define HUTOKEN_CORE_H

#ifdef USE_FOMA
#include "Python.h"
#include "fomalib.h"
#endif

#include "hutoken/taskqueue.h"

void encode(struct EncodeTask* task);
void decode(struct DecodeTask* task);
#ifdef USE_FOMA
PyObject* initialize_foma(void);
PyObject* look_up_word(struct apply_handle* handle,
                       char* word,
                       bool only_longest);
#endif

#endif
