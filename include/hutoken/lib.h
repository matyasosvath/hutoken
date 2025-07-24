#ifndef HUTOKEN_LIB_H
#define HUTOKEN_LIB_H

#include "Python.h"

PyObject* p_bpe_train(PyObject* self, PyObject* args);
PyObject* p_bbpe_train(PyObject* self, PyObject* args);
PyObject* p_encode(PyObject* self, PyObject* args);
PyMODINIT_FUNC PyInit__hutoken(void);
PyObject* p_initialize_foma(PyObject* self);
PyObject* p_look_up_word(PyObject* self, PyObject* args);

#endif
