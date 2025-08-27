#include "hutoken/arena.h"

#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

static const size_t ARENA_DEFAULT_ALIGNMENT = alignof(max_align_t);

static inline size_t align_up(const size_t value, const size_t alignment);

bool arena_create(struct Arena* arena, const size_t size) {
    if (arena == NULL || size == 0) {
        return false;
    }

    arena->buffer = malloc(size);
    if (arena->buffer == NULL) {
        return false;
    }

    arena->total_size = size;
    arena->current_offset = 0;

    return true;
}

void arena_destroy(struct Arena* arena) {
    if (arena != NULL && arena->buffer != NULL) {
        free(arena->buffer);
        arena->buffer = NULL;
        arena->total_size = 0;
        arena->current_offset = 0;
    }
}

void* arena_alloc(struct Arena* arena, const size_t size) {
    if (arena == NULL || size == 0) {
        return NULL;
    }

    size_t aligned_offset =
        align_up(arena->current_offset, ARENA_DEFAULT_ALIGNMENT);

    if (aligned_offset + size > arena->total_size) {
        return NULL;
    }

    void* ptr = arena->buffer + aligned_offset;

    arena->current_offset = aligned_offset + size;

    return ptr;
}

void arena_reset(struct Arena* arena) {
    if (arena != NULL) {
        arena->current_offset = 0;
    }
}

static inline size_t align_up(const size_t value, const size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}
