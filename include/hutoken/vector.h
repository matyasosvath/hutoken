#ifndef HUTOKEN_VECTOR_H
#define HUTOKEN_VECTOR_H

#include <stddef.h>

struct IntVector {
    int* data;
    size_t size;
    size_t capacity;
};

void vector_init(struct IntVector* vec, size_t initial_capacity);

void vector_push(struct IntVector* vec, int value);

void vector_append_array(struct IntVector* vec,
                         const int* values,
                         size_t count);

void vector_free(struct IntVector* vec);

#endif
