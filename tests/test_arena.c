#include <assert.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/arena.h"

#define RUN_TEST(test)                          \
    do {                                        \
        printf("Running test: %s...\n", #test); \
        test();                                 \
    } while (0)

void test_arena_create_success(void) {
    struct Arena arena;
    const size_t size = 1024;

    assert(arena_create(&arena, size) == true);
    assert(arena.buffer != NULL);
    assert(arena.total_size == size);
    assert(arena.current_offset == 0);

    arena_destroy(&arena);
}

void test_arena_create_zero_size(void) {
    struct Arena arena;
    assert(arena_create(&arena, 0) == false);
}

void test_arena_create_null_arena(void) {
    assert(arena_create(NULL, 1024) == false);
}

void test_arena_alloc_simple(void) {
    struct Arena arena;
    arena_create(&arena, 1024);

    void* ptr = arena_alloc(&arena, 100);
    assert(ptr != NULL);
    assert(arena.current_offset >= 100);

    arena_destroy(&arena);
}

void test_arena_alloc_multiple(void) {
    struct Arena arena;
    arena_create(&arena, 1024);

    void* ptr1 = arena_alloc(&arena, 10);
    assert(ptr1 != NULL);
    size_t offset1 = arena.current_offset;

    void* ptr2 = arena_alloc(&arena, 20);
    assert(ptr2 != NULL);
    size_t offset2 = arena.current_offset;

    assert(offset2 > offset1);
    assert(offset2 - offset1 >= 20);

    arena_destroy(&arena);
}

void test_arena_alloc_full(void) {
    struct Arena arena;
    const size_t size = 128;
    arena_create(&arena, size);

    void* ptr = arena_alloc(&arena, size);
    assert(ptr != NULL);
    assert(arena.current_offset >= size);

    arena_destroy(&arena);
}

void test_arena_alloc_out_of_memory(void) {
    struct Arena arena;
    arena_create(&arena, 50);

    void* ptr1 = arena_alloc(&arena, 40);
    assert(ptr1 != NULL);

    void* ptr2 = arena_alloc(&arena, 20);
    assert(ptr2 == NULL);

    arena_destroy(&arena);
}

void test_arena_alloc_zero_size(void) {
    struct Arena arena;
    arena_create(&arena, 1024);

    void* ptr = arena_alloc(&arena, 0);
    assert(ptr == NULL);

    arena_destroy(&arena);
}

void test_arena_alloc_null_arena(void) {
    void* ptr = arena_alloc(NULL, 100);
    assert(ptr == NULL);
}

void test_arena_alignment(void) {
    struct Arena arena;
    arena_create(&arena, 1024);

    arena_alloc(&arena, 1);

    int* int_ptr = (int*)arena_alloc(&arena, sizeof(int));
    assert(int_ptr != NULL);
    assert((uintptr_t)int_ptr % alignof(int) == 0);

    double* double_ptr = (double*)arena_alloc(&arena, sizeof(double));
    assert(double_ptr != NULL);
    assert((uintptr_t)double_ptr % alignof(double) == 0);

    arena_destroy(&arena);
}

void test_arena_alloc_write_and_read(void) {
    struct Arena arena;
    arena_create(&arena, 256);

    char* str = (char*)arena_alloc(&arena, 12);
    assert(str != NULL);
    strcpy(str, "hello arena");
    assert(strcmp(str, "hello arena") == 0);

    int* numbers = (int*)arena_alloc(&arena, 5 * sizeof(int));
    assert(numbers != NULL);
    for (int i = 0; i < 5; i++) {
        numbers[i] = i * 10;
    }

    assert(numbers[0] == 0);
    assert(numbers[4] == 40);

    arena_destroy(&arena);
}

void test_arena_reset(void) {
    struct Arena arena;
    arena_create(&arena, 1024);

    void* ptr1 = arena_alloc(&arena, 500);
    assert(ptr1 != NULL);
    assert(arena.current_offset > 0);

    arena_reset(&arena);
    assert(arena.current_offset == 0);

    void* ptr2 = arena_alloc(&arena, 800);
    assert(ptr2 != NULL);
    assert(arena.current_offset >= 800);

    assert((uintptr_t)ptr1 == (uintptr_t)ptr2);

    arena_destroy(&arena);
}

void test_arena_destroy_null_arena(void) {
    arena_destroy(NULL);
    assert(true);
}

void test_arena_reset_null_arena(void) {
    arena_reset(NULL);
    assert(true);
}

int main(void) {
    puts("Starting arena tests.\n");

    RUN_TEST(test_arena_create_success);
    RUN_TEST(test_arena_create_zero_size);
    RUN_TEST(test_arena_create_null_arena);
    RUN_TEST(test_arena_alloc_simple);
    RUN_TEST(test_arena_alloc_multiple);
    RUN_TEST(test_arena_alloc_full);
    RUN_TEST(test_arena_alloc_out_of_memory);
    RUN_TEST(test_arena_alloc_zero_size);
    RUN_TEST(test_arena_alloc_null_arena);
    RUN_TEST(test_arena_alignment);
    RUN_TEST(test_arena_alloc_write_and_read);
    RUN_TEST(test_arena_reset);
    RUN_TEST(test_arena_destroy_null_arena);
    RUN_TEST(test_arena_reset_null_arena);

    puts("\nAll arena tests passed successfully!");

    return EXIT_SUCCESS;
}
