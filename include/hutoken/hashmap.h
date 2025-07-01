#ifndef HUTOKEN_HASHMAP_H
#define HUTOKEN_HASHMAP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct HashMap {
    size_t element_size;
    size_t capacity;
    size_t bucket_size;
    size_t bucket_num;
    size_t count;
    size_t mask;
    size_t growat;
    size_t shrinkat; // hash map shrinks if number of items falls below threshold
    uint8_t loadfactor; // threshold of filled buckets before resizing
    uint8_t growpower; // how much the hash map will grow when resized
    bool oom; // signal that last called failed due to system being out of memory.
    void *buckets;
    void *spare;
    void *edata; // temp data for a bucket
};

struct HashMap *hashmap_new(size_t capacity);
const void *hashmap_set(struct HashMap *map, const void *item);
int hashmap_get(struct HashMap *map, const void *key);
const void *hashmap_delete(struct HashMap *map, const void *key);
void hashmap_clear(struct HashMap *map, bool update_cap);
bool hashmap_iter(struct HashMap *map, size_t *i, void **item);
void hashmap_free(struct HashMap *map);

struct Token {
    char *key;
    int value;
};

#endif
