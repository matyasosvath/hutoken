#ifndef HUTOKEN_HASH_H
#define HUTOKEN_HASH_H

#include <stddef.h>
#include <stdint.h>

#ifndef SEED
#define SEED 42
#endif

uint64_t hashmap_murmur(const void* data, size_t len);

#endif
