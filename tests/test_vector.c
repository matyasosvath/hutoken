#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/vector.h"

#define RUN_TEST(test)                          \
    do {                                        \
        printf("Running test: %s...\n", #test); \
        test();                                 \
    } while (0)

void test_vector_init(void) {
    struct IntVector vec;
    vector_init(&vec, 10);

    assert(vec.data != NULL);
    assert(vec.size == 0);
    assert(vec.capacity == 10);

    vector_free(&vec);
}

void test_vector_init_zero_capacity(void) {
    struct IntVector vec;
    vector_init(&vec, 0);

    assert(vec.data != NULL);
    assert(vec.size == 0);
    assert(vec.capacity > 0);

    vector_free(&vec);
}

void test_vector_push_no_grow(void) {
    struct IntVector vec;
    vector_init(&vec, 5);

    for (int i = 0; i < 5; ++i) {
        vector_push(&vec, i * 10);
    }

    assert(vec.size == 5);
    assert(vec.capacity == 5);
    assert(vec.data[0] == 0);
    assert(vec.data[4] == 40);

    vector_free(&vec);
}

void test_vector_push_with_grow(void) {
    struct IntVector vec;
    vector_init(&vec, 4);

    for (int i = 0; i < 10; ++i) {
        vector_push(&vec, i);
    }

    assert(vec.size == 10);
    assert(vec.capacity >= 10);
    for (int i = 0; i < 10; ++i) {
        assert(vec.data[i] == i);
    }

    vector_free(&vec);
}

void test_vector_append_no_grow(void) {
    struct IntVector vec;
    vector_init(&vec, 20);
    int new_data[] = {10, 20, 30, 40, 50};

    vector_append_array(&vec, new_data, 5);

    assert(vec.size == 5);
    assert(vec.capacity == 20);
    assert(memcmp(vec.data, new_data, 5 * sizeof(int)) == 0);

    vector_free(&vec);
}

void test_vector_append_with_grow(void) {
    struct IntVector vec;
    vector_init(&vec, 2);
    vector_push(&vec, 1);  // size = 1
    int new_data[] = {10, 20, 30, 40, 50};

    vector_append_array(&vec, new_data, 5);

    assert(vec.size == 6);
    assert(vec.capacity >= 6);
    assert(vec.data[0] == 1);
    assert(vec.data[1] == 10);
    assert(vec.data[5] == 50);

    vector_free(&vec);
}

void test_vector_append_large_array(void) {
    struct IntVector vec;
    vector_init(&vec, 0);
    int large_data[100];
    for (int i = 0; i < 100; ++i) {
        large_data[i] = i;
    }

    vector_append_array(&vec, large_data, 100);

    assert(vec.size == 100);
    assert(vec.capacity >= 100);
    assert(memcmp(vec.data, large_data, 100 * sizeof(int)) == 0);

    vector_free(&vec);
}

void test_vector_ops_on_null(void) {
    int data[] = {1};
    vector_init(NULL, 10);
    vector_push(NULL, 1);
    vector_append_array(NULL, data, 1);
    vector_free(NULL);
}

void test_vector_append_invalid_args(void) {
    struct IntVector vec;
    vector_init(&vec, 10);
    vector_push(&vec, 99);

    vector_append_array(&vec, NULL, 5);
    assert(vec.size == 1);

    int data[] = {1, 2, 3};
    vector_append_array(&vec, data, 0);
    assert(vec.size == 1);
    assert(vec.data[0] == 99);

    vector_free(&vec);
}

void test_vector_mixed_ops(void) {
    struct IntVector vec;
    vector_init(&vec, 3);
    vector_push(&vec, 1);
    vector_push(&vec, 2);

    int data1[] = {3, 4, 5};
    vector_append_array(&vec, data1, 3);

    assert(vec.size == 5);
    assert(vec.data[4] == 5);

    int data2[] = {6, 7, 8, 9, 10};
    vector_append_array(&vec, data2, 5);

    assert(vec.size == 10);
    assert(vec.data[9] == 10);

    vector_free(&vec);
    assert(vec.data == NULL);
    assert(vec.size == 0);
    assert(vec.capacity == 0);
}

int main(void) {
    puts("Starting IntVector tests.\n");

    RUN_TEST(test_vector_init);
    RUN_TEST(test_vector_init_zero_capacity);
    RUN_TEST(test_vector_push_no_grow);
    RUN_TEST(test_vector_push_with_grow);
    RUN_TEST(test_vector_append_no_grow);
    RUN_TEST(test_vector_append_with_grow);
    RUN_TEST(test_vector_append_large_array);
    RUN_TEST(test_vector_ops_on_null);
    RUN_TEST(test_vector_append_invalid_args);
    RUN_TEST(test_vector_mixed_ops);

    puts("\nAll IntVector tests passed successfully!");

    return EXIT_SUCCESS;
}
