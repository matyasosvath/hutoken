#include "hutoken/vector.h"

#include <stdlib.h>
#include <string.h>

#include "hutoken/helper.h"

static void vector_grow(struct IntVector* vec, size_t min_capacity);

void vector_init(struct IntVector* vec, size_t initial_capacity) {
    if (!vec) {
        return;
    }
    if (initial_capacity == 0) {
        initial_capacity = 8;
    }
    vec->data = (int*)malloc(initial_capacity * sizeof(int));
    if (!vec->data) {
        log_debug("Error: Failed to allocate memory for dynamic array.");
        vec->size = 0;
        vec->capacity = 0;
        return;
    }
    vec->size = 0;
    vec->capacity = initial_capacity;
}

void vector_push(struct IntVector* vec, int value) {
    if (!vec || !vec->data) {
        return;
    }
    if (vec->size >= vec->capacity) {
        vector_grow(vec, vec->capacity + 1);
        if (vec->size >= vec->capacity) {
            return;
        }
    }
    vec->data[vec->size++] = value;
}

void vector_append_array(struct IntVector* vec,
                         const int* values,
                         size_t count) {
    if (!vec || !vec->data || !values || count == 0) {
        return;
    }
    if (vec->size + count > vec->capacity) {
        vector_grow(vec, vec->size + count);
        if (vec->size + count > vec->capacity) {
            return;
        }
    }
    memcpy(vec->data + vec->size, values, count * sizeof(int));
    vec->size += count;
}

void vector_free(struct IntVector* vec) {
    if (vec && vec->data) {
        free(vec->data);
        vec->data = NULL;
        vec->size = 0;
        vec->capacity = 0;
    }
}

static void vector_grow(struct IntVector* vec, size_t min_capacity) {
    size_t new_capacity = vec->capacity * 2;
    if (new_capacity < min_capacity) {
        new_capacity = min_capacity;
    }
    int* new_data = (int*)realloc(vec->data, new_capacity * sizeof(int));
    if (!new_data) {
        log_debug("Error: Failed to reallocate memory for dynamic array.");
        return;
    }
    vec->data = new_data;
    vec->capacity = new_capacity;
}
