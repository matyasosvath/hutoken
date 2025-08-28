#include "hutoken/hashmap.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/bpe.h"
#include "hutoken/hash.h"

#define GROW_AT 0.60             /* 60% */
#define SHRINK_AT 0.10           /* 10% */
#define HASHMAP_LOAD_FACTOR 0.60 /* 60% */

struct Bucket {
    uint64_t hash : 48;
    uint64_t dib : 16;
};

// Check for NaN and clamp between 50% and 90%
static double clamp_load_factor(double factor, double default_factor) {
    return factor != factor ? default_factor
           : factor < 0.50  ? 0.50
           : factor > 0.95  ? 0.95
                            : factor;
}

// Calculate memory address of the i-th bucket
static struct Bucket* bucket_at(struct HashMap* map, size_t index) {
    void* buckets = map->buckets;
    size_t bucket_size = map->bucket_size;
    return (struct Bucket*)(((char*)buckets) + (bucket_size * index));
}

// Calculate memory address of the i-th bucket's token data
static void* bucket_item(struct Bucket* entry) {
    return ((char*)entry) + sizeof(struct Bucket);
}

// Clip hash value to 48 bits (ensuring it stays within the bucket limit)
static uint64_t clip_hash(uint64_t hash) {
    return hash & 0xFFFFFFFFFFFF;
}

static uint64_t get_hash(struct HashMap* map, const void* item) {
    const struct Token* token = item;
    uint64_t hash = hashmap_murmur(token->key, strlen(token->key));
    return clip_hash(hash);
}

struct HashMap* hashmap_new(size_t capacity,
                            size_t element_size,
                            uint64_t (*hash_func)(const void* item),
                            int (*compare_func)(const void* a, const void* b)) {
    size_t ncap = 16;

    if (capacity < ncap) {
        capacity = ncap;
    } else {
        while (ncap < capacity) {
            ncap *= 2;
        }
        capacity = ncap;
    }

    size_t bucket_size = sizeof(struct Bucket) + element_size;

    // platform dependent alignment
    while (bucket_size & (sizeof(uintptr_t) - 1)) {
        bucket_size++;
    }

    // allocate space for hashmap, plus two extra for spare and edata
    size_t size = sizeof(struct HashMap) + bucket_size * 2;

    struct HashMap* map = malloc(size);

    if (!map) {
        return NULL;
    }

    memset(map, 0, sizeof(struct HashMap));

    map->element_size = element_size;
    map->bucket_size = bucket_size;
    map->capacity = capacity;
    map->bucket_num = capacity;
    map->mask = map->bucket_num - 1;
    map->hash_func = hash_func;
    map->compare_func = compare_func;

    map->buckets = malloc(map->bucket_size * map->bucket_num);

    if (!map->buckets) {
        free(map);
        return NULL;
    }

    memset(map->buckets, 0, map->bucket_size * map->bucket_num);

    map->growpower = 1;
    map->loadfactor = clamp_load_factor(HASHMAP_LOAD_FACTOR, SEED) * 100;
    map->growat = map->bucket_num * (map->loadfactor / 100.0);
    map->shrinkat = map->bucket_num * SHRINK_AT;

    map->spare = ((char*)map) + sizeof(struct HashMap);
    map->edata = (char*)map->spare + bucket_size;

    return map;
}

static bool resize(struct HashMap* map, size_t new_cap) {
    struct HashMap* map2 = hashmap_new(new_cap, map->element_size,
                                       map->hash_func, map->compare_func);

    if (!map2) {
        return false;
    }

    for (size_t i = 0; i < map->bucket_num; i++) {
        struct Bucket* entry = bucket_at(map, i);

        // no item found
        if (!entry->dib) {
            continue;
        }

        entry->dib = 1;

        size_t j = entry->hash & map2->mask;

        while (true) {
            struct Bucket* bucket = bucket_at(map2, j);

            if (bucket->dib == 0) {
                memcpy(bucket, entry, map->bucket_size);
                break;
            }

            if (bucket->dib < entry->dib) {
                memcpy(map2->spare, bucket, map->bucket_size);
                memcpy(bucket, entry, map->bucket_size);
                memcpy(entry, map2->spare, map->bucket_size);
            }

            j = (j + 1) & map2->mask;

            entry->dib += 1;
        }
    }

    free(map->buckets);

    map->buckets = map2->buckets;
    map->bucket_num = map2->bucket_num;
    map->mask = map2->mask;
    map->growat = map2->growat;
    map->shrinkat = map2->shrinkat;

    free(map2);

    return true;
}

// Insert (or replace) an item. Replaced item is returned, NULL otherwise
// Operation may allocate memory. If unable to allocate more memory, then
// NULL is returned and map->oom will be true.
const void* hashmap_set(struct HashMap* map, const void* item) {
    uint64_t hash = map->hash_func(item);
    hash = clip_hash(hash);

    map->oom = false;

    if (map->count >= map->growat) {
        if (!resize(map, map->bucket_num * (1 << map->growpower))) {
            map->oom = true;
            return NULL;
        }
    }

    struct Bucket* entry = map->edata;  // temp bucket

    entry->hash = hash;
    entry->dib = 1;

    void* eitem = bucket_item(entry);  // token

    // copy item (token) in entry bucket
    memcpy(eitem, item, map->element_size);

    void* bitem = NULL;

    // determine initial index based on the hash and the map’s mask
    // (a bit mask that confines the index to the valid range of bucket array)
    size_t i = entry->hash & map->mask;

    // start probing loop for insertion
    while (1) {
        struct Bucket* bucket = bucket_at(map, i);

        if (bucket->dib == 0) {  // unoccupied bucket, entry can be inserted

            // copy entry (hash, dib and eitem (item)) into bucket
            memcpy(bucket, entry, map->bucket_size);
            map->count++;
            return NULL;
        }

        bitem = bucket_item(bucket);  // token

        // replace item, if relevant
        if (entry->hash == bucket->hash &&
            (map->compare_func(eitem, bitem) == 0)) {
            memcpy(map->spare, bitem, map->element_size);
            memcpy(bitem, eitem, map->element_size);

            return map->spare;
        }

        // collision handling (with robin hood hashing)
        if (bucket->dib < entry->dib) {
            memcpy(map->spare, bucket, map->bucket_size);
            memcpy(bucket, entry, map->bucket_size);
            memcpy(entry, map->spare, map->bucket_size);

            eitem = bucket_item(entry);
        }

        i = (i + 1) & map->mask;

        entry->dib += 1;
    }
}

// return item based on key. If item is not found, NULL is returned.
void* hashmap_get(struct HashMap* map, const void* key) {
    uint64_t hash = get_hash(map, key);

    hash = clip_hash(hash);

    // only lower bits of the hash are used in index calculation
    // creating an index in the range [0, bucket_num - 1]
    size_t i = hash & map->mask;

    while (true) {
        struct Bucket* bucket = bucket_at(map, i);

        if (!bucket->dib) {
            return NULL;
        }

        if (bucket->hash == hash) {
            void* bitem = bucket_item(bucket);

            if (map->compare_func(key, bitem) == false) {
                return bitem;
            }
        }
        i = (i + 1) & map->mask;
    }
}

// Get key based on value. If item is not found, NULL is returned.
// NOTE: This function is specific to the `struct Token` vocab map.
char* hashmap_get_key(struct HashMap* map, const int value) {
    size_t i = 0;
    void* item = NULL;

    while (hashmap_iter(map, &i, &item)) {
        struct Token* token = item;
        if (!token) {
            continue;
        }
        if (token->value == value) {
            return token->key;
        }
    }

    return NULL;
}

// Remove item from the hash map and return it.
// If item not found, NULL is returned.
const void* hashmap_delete(struct HashMap* map, const void* key) {
    uint64_t hash = map->hash_func(key);
    hash = clip_hash(hash);

    map->oom = false;

    size_t i = hash & map->mask;

    while (1) {
        struct Bucket* bucket = bucket_at(map, i);

        if (!bucket->dib) {
            return NULL;
        }

        void* bitem = bucket_item(bucket);  // token

        if (bucket->hash == hash && (map->compare_func(key, bitem) == false)) {
            memcpy(map->spare, bitem, map->element_size);

            bucket->dib = 0;

            while (true) {
                struct Bucket* prev = bucket;

                i = (i + 1) & map->mask;

                bucket = bucket_at(map, i);

                if (bucket->dib <= 1) {
                    prev->dib = 0;
                    break;
                }

                memcpy(prev, bucket, map->bucket_size);

                prev->dib--;
            }

            map->count--;

            if (map->bucket_num > map->capacity &&
                map->count <= map->shrinkat) {
                // Ignore the return value. It's ok for the resize operation to
                // fail to allocate enough memory because a shrink operation
                // does not change the integrity of the data.
                resize(map, map->bucket_num / 2);
            }
            return map->spare;
        }
        i = (i + 1) & map->mask;
    }
}

// Clear hashmap. When update_cap is true, the map's
// capacity will match the current number of buckets.
void hashmap_clear(struct HashMap* map, bool update_cap) {
    map->count = 0;

    if (update_cap) {
        map->capacity = map->bucket_num;
    } else if (map->bucket_num != map->capacity) {
        void* new_buckets = malloc(map->bucket_size * map->capacity);

        if (new_buckets) {
            free(map->buckets);
            map->buckets = new_buckets;
        }

        map->bucket_num = map->capacity;
    }

    // clears memory, removing all buckets
    (void)memset(map->buckets, 0, map->bucket_size * map->bucket_num);

    map->mask = map->bucket_num - 1;
    map->growat = map->bucket_num * (map->loadfactor / 100.0);
    map->shrinkat = map->bucket_num * SHRINK_AT;
}

// iterate over hashmap
bool hashmap_iter(struct HashMap* map, size_t* i, void** item) {
    struct Bucket* bucket = NULL;

    do {
        if (*i >= map->bucket_num) {
            return false;
        }

        bucket = bucket_at(map, *i);

        (*i)++;

    } while (!bucket->dib);

    *item = bucket_item(bucket);

    return true;
}

// Free hashmap
void hashmap_free(struct HashMap* map) {
    if (!map) {
        return;
    }
    free(map->buckets);
    free(map);
}
