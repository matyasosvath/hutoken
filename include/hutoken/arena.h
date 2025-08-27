#ifndef HUTOKEN_ARENA_H
#define HUTOKEN_ARENA_H

#include <stdbool.h>
#include <stddef.h>

struct Arena {
    unsigned char* buffer;
    size_t total_size;
    size_t current_offset;
};

bool arena_create(struct Arena* arena, const size_t size);
void arena_destroy(struct Arena* arena);
void* arena_alloc(struct Arena* arena, size_t size);
void arena_reset(struct Arena* arena);

#endif
