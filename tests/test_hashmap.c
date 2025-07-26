#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "hashmap.h"

int main() {
    struct HashMap* map = hashmap_new(16);

    struct Token token1 = {.key = "key1", .value = 1};
    struct Token token2 = {.key = "key2", .value = 2};
    struct Token token3 = {.key = "key3", .value = 3};

    assert(hashmap_set(map, &token1) == NULL);
    assert(hashmap_set(map, &token2) == NULL);
    assert(hashmap_set(map, &token3) == NULL);

    struct Token* result = (struct Token*)hashmap_get(map, &token1);
    assert(result && result->value == 1);

    result = (struct Token*)hashmap_get(map, &token2);
    assert(result && result->value == 2);

    result = (struct Token*)hashmap_get(map, &token3);
    assert(result && result->value == 3);

    assert(hashmap_count(map) == 3);

    assert(hashmap_delete(map, &token2) != NULL);
    assert(hashmap_get(map, &token2) == NULL);
    assert(hashmap_count(map) == 2);

    hashmap_clear(map, false);
    assert(hashmap_count(map) == 0);

    hashmap_free(map);

    printf("All tests passed!\n");
    return 0;
}