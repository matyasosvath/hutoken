#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "hutoken/queue.h"

#define RUN_TEST(test)                          \
    do {                                        \
        printf("Running test: %s...\n", #test); \
        test();                                 \
    } while (0)

void test_init(void) {
    struct MinPQ pq;

    assert(min_pq_init(&pq, 10) == MIN_PQ_SUCCESS);
    assert(pq.data != NULL);
    assert(pq.size == 0);
    assert(pq.capacity == 10);
    assert(min_pq_size(&pq) == 0);
    assert(min_pq_is_empty(&pq) == true);

    min_pq_release(&pq);
}

void test_init_with_zero_capacity(void) {
    struct MinPQ pq;

    assert(min_pq_init(&pq, 0) == MIN_PQ_SUCCESS);
    assert(pq.data != NULL);
    assert(pq.size == 0);
    assert(pq.capacity > 0);
    assert(min_pq_is_empty(&pq) == true);

    min_pq_release(&pq);
}

void test_push_single_element(void) {
    struct MinPQ pq;
    min_pq_init(&pq, 4);

    struct MergeCandidate candidate = {
        .rank = 10, .left_idx = 0, .right_idx = 1};
    assert(min_pq_push(&pq, candidate) == MIN_PQ_SUCCESS);

    assert(min_pq_size(&pq) == 1);
    assert(min_pq_is_empty(&pq) == false);

    min_pq_release(&pq);
}

void test_pop_from_empty_queue(void) {
    struct MinPQ pq;
    min_pq_init(&pq, 4);

    struct MergeCandidate candidate;
    assert(min_pq_pop(&pq, &candidate) == MIN_PQ_EMPTY);
    assert(min_pq_is_empty(&pq) == true);

    min_pq_release(&pq);
}

void test_push_then_pop_single_element(void) {
    struct MinPQ pq;
    min_pq_init(&pq, 4);

    struct MergeCandidate pushed = {.rank = 10, .left_idx = 0, .right_idx = 1};
    min_pq_push(&pq, pushed);

    struct MergeCandidate popped;
    assert(min_pq_pop(&pq, &popped) == MIN_PQ_SUCCESS);

    assert(min_pq_is_empty(&pq) == true);
    assert(popped.rank == 10);
    assert(popped.left_idx == 0);
    assert(popped.right_idx == 1);

    min_pq_release(&pq);
}

void test_pop_order_is_correct(void) {
    struct MinPQ pq;
    min_pq_init(&pq, 10);

    struct MergeCandidate c1 = {.rank = 20, .left_idx = 1, .right_idx = 2};
    struct MergeCandidate c2 = {.rank = 5, .left_idx = 2, .right_idx = 3};
    struct MergeCandidate c3 = {.rank = 15, .left_idx = 3, .right_idx = 4};
    struct MergeCandidate c4 = {.rank = 10, .left_idx = 4, .right_idx = 5};

    min_pq_push(&pq, c1);
    min_pq_push(&pq, c2);
    min_pq_push(&pq, c3);
    min_pq_push(&pq, c4);

    assert(min_pq_size(&pq) == 4);

    struct MergeCandidate popped;
    int last_rank = -1;

    assert(min_pq_pop(&pq, &popped) == MIN_PQ_SUCCESS);
    assert(popped.rank == 5);
    last_rank = popped.rank;

    assert(min_pq_pop(&pq, &popped) == MIN_PQ_SUCCESS);
    assert(popped.rank >= last_rank);
    assert(popped.rank == 10);
    last_rank = popped.rank;

    assert(min_pq_pop(&pq, &popped) == MIN_PQ_SUCCESS);
    assert(popped.rank >= last_rank);
    assert(popped.rank == 15);
    last_rank = popped.rank;

    assert(min_pq_pop(&pq, &popped) == MIN_PQ_SUCCESS);
    assert(popped.rank >= last_rank);
    assert(popped.rank == 20);

    assert(min_pq_is_empty(&pq) == true);

    min_pq_release(&pq);
}

void test_push_triggers_resize(void) {
    struct MinPQ pq;
    min_pq_init(&pq, 2);
    assert(pq.capacity == 2);

    struct MergeCandidate c1 = {.rank = 10};
    struct MergeCandidate c2 = {.rank = 20};
    struct MergeCandidate c3 = {.rank = 5};

    assert(min_pq_push(&pq, c1) == MIN_PQ_SUCCESS);
    assert(min_pq_push(&pq, c2) == MIN_PQ_SUCCESS);
    assert(min_pq_push(&pq, c3) == MIN_PQ_SUCCESS);

    assert(pq.capacity > 2);
    assert(min_pq_size(&pq) == 3);

    struct MergeCandidate popped;
    min_pq_pop(&pq, &popped);
    assert(popped.rank == 5);

    min_pq_release(&pq);
}

void test_duplicate_ranks(void) {
    struct MinPQ pq;
    min_pq_init(&pq, 5);

    struct MergeCandidate c1 = {.rank = 10, .left_idx = 1};
    struct MergeCandidate c2 = {.rank = 5, .left_idx = 2};
    struct MergeCandidate c3 = {.rank = 10, .left_idx = 3};
    struct MergeCandidate c4 = {.rank = 5, .left_idx = 4};

    min_pq_push(&pq, c1);
    min_pq_push(&pq, c2);
    min_pq_push(&pq, c3);
    min_pq_push(&pq, c4);

    assert(min_pq_size(&pq) == 4);

    struct MergeCandidate popped;
    min_pq_pop(&pq, &popped);
    assert(popped.rank == 5);

    min_pq_pop(&pq, &popped);
    assert(popped.rank == 5);

    min_pq_pop(&pq, &popped);
    assert(popped.rank == 10);

    min_pq_pop(&pq, &popped);
    assert(popped.rank == 10);

    assert(min_pq_is_empty(&pq) == true);

    min_pq_release(&pq);
}

void test_invalid_arguments(void) {
    struct MergeCandidate candidate = {.rank = 10, .left_idx = 1};

    assert(min_pq_push(NULL, candidate) == MIN_PQ_INVALID_ARGUMENT);
    assert(min_pq_pop(NULL, &candidate) == MIN_PQ_INVALID_ARGUMENT);
    assert(min_pq_size(NULL) == 0);
    assert(min_pq_is_empty(NULL) == true);
    min_pq_release(NULL);
}

int main(void) {
    puts("Starting queue tests.\n");

    RUN_TEST(test_init);
    RUN_TEST(test_init_with_zero_capacity);
    RUN_TEST(test_push_single_element);
    RUN_TEST(test_pop_from_empty_queue);
    RUN_TEST(test_push_then_pop_single_element);
    RUN_TEST(test_pop_order_is_correct);
    RUN_TEST(test_push_triggers_resize);
    RUN_TEST(test_duplicate_ranks);
    RUN_TEST(test_invalid_arguments);

    puts("\nAll tests passed successfully!");
    return EXIT_SUCCESS;
}
