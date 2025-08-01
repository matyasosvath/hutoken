#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../include/hutoken/bpe.h"
#include "../include/hutoken/hashmap.h"

int main(void) {
    struct HashMap* map = hashmap_new(16);
    assert(map && map->capacity == 16 && map->count == 0);

    struct Token token1 = {.key = "key1", .value = 1};
    struct Token token2 = {.key = "key2", .value = 2};
    struct Token token3 = {.key = "key3", .value = 3};

    assert(hashmap_set(map, &token1) == NULL);
    assert(hashmap_set(map, &token2) == NULL);
    assert(hashmap_set(map, &token3) == NULL);

    int result = hashmap_get(map, &token1);
    assert(result == 1);

    result = hashmap_get(map, &token2);
    assert(result == 2);

    result = hashmap_get(map, &token3);
    assert(result == 3);

    assert(map->count == 3);

    assert(hashmap_delete(map, &token2) != NULL);
    assert(hashmap_get(map, &token2) == -1);
    assert(map->count == 2);

    char* key_result = hashmap_get_key(map, 1);
    assert(key_result && strcmp(key_result, "key1") == 0);
    key_result = hashmap_get_key(map, 2);
    assert(key_result == NULL);

    hashmap_clear(map, false);
    assert(map->count == 0);

    hashmap_free(map);

    printf("All tests passed!\n");
    return 0;
}
