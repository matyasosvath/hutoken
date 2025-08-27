#include "hutoken/queue.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/arena.h"

static const int INITIAL_CAPACITY = 16;

static enum MinPQError resize(struct MinPQ* pq, const size_t new_capacity);
static enum MinPQError resize_arena(struct Arena* arena,
                                    struct MinPQ* pq,
                                    const size_t new_capacity);
static void heapify_up(struct MinPQ* pq, const size_t index);
static void heapify_down(struct MinPQ* pq, const size_t index);
static void swap(struct MergeCandidate* a, struct MergeCandidate* b);

enum MinPQError min_pq_init(struct MinPQ* pq, const size_t capacity) {
    size_t initial_cap = (capacity > 0) ? capacity : INITIAL_CAPACITY;
    pq->data = malloc(initial_cap * sizeof(struct MergeCandidate));
    if (!pq->data) {
        free(pq);
        return MIN_PQ_ALLOC_ERROR;
    }

    pq->size = 0;
    pq->capacity = initial_cap;

    return MIN_PQ_SUCCESS;
}

void min_pq_release(struct MinPQ* pq) {
    if (!pq) {
        return;
    }

    free(pq->data);
}

enum MinPQError min_pq_push(struct MinPQ* pq,
                            const struct MergeCandidate candidate) {
    if (!pq) {
        return MIN_PQ_INVALID_ARGUMENT;
    }

    if (pq->size >= pq->capacity) {
        if (resize(pq, pq->capacity * 2) != MIN_PQ_SUCCESS) {
            return MIN_PQ_ALLOC_ERROR;
        }
    }

    pq->data[pq->size] = candidate;
    heapify_up(pq, pq->size);
    ++pq->size;

    return MIN_PQ_SUCCESS;
}

enum MinPQError min_pq_pop(struct MinPQ* pq, struct MergeCandidate* candidate) {
    if (!pq) {
        return MIN_PQ_INVALID_ARGUMENT;
    }

    if (pq->size == 0) {
        return MIN_PQ_EMPTY;
    }

    *candidate = pq->data[0];
    pq->data[0] = pq->data[pq->size - 1];
    --pq->size;
    heapify_down(pq, 0);

    return MIN_PQ_SUCCESS;
}

size_t min_pq_size(const struct MinPQ* pq) {
    return pq ? pq->size : 0;
}

bool min_pq_is_empty(const struct MinPQ* pq) {
    return pq ? pq->size == 0 : true;
}

enum MinPQError min_pq_init_arena(struct Arena* arena,
                                  struct MinPQ* pq,
                                  const size_t capacity) {
    size_t initial_cap = (capacity > 0) ? capacity : INITIAL_CAPACITY;
    pq->data = arena_alloc(arena, initial_cap * sizeof(struct MergeCandidate));
    if (!pq->data) {
        return MIN_PQ_ALLOC_ERROR;
    }

    pq->size = 0;
    pq->capacity = initial_cap;

    return MIN_PQ_SUCCESS;
}

enum MinPQError min_pq_push_arena(struct Arena* arena,
                                  struct MinPQ* pq,
                                  const struct MergeCandidate candidate) {
    if (!pq) {
        return MIN_PQ_INVALID_ARGUMENT;
    }

    if (pq->size >= pq->capacity) {
        if (resize_arena(arena, pq, pq->capacity * 2) != MIN_PQ_SUCCESS) {
            return MIN_PQ_ALLOC_ERROR;
        }
    }

    pq->data[pq->size] = candidate;
    heapify_up(pq, pq->size);
    ++pq->size;

    return MIN_PQ_SUCCESS;
}

static enum MinPQError resize(struct MinPQ* pq, const size_t new_capacity) {
    struct MergeCandidate* new_data =
        realloc(pq->data, new_capacity * sizeof(struct MergeCandidate));

    if (!new_data) {
        return MIN_PQ_ALLOC_ERROR;
    }

    pq->data = new_data;
    pq->capacity = new_capacity;

    return MIN_PQ_SUCCESS;
}

static enum MinPQError resize_arena(struct Arena* arena,
                                    struct MinPQ* pq,
                                    const size_t new_capacity) {
    struct MergeCandidate* new_data =
        arena_alloc(arena, new_capacity * sizeof(struct MergeCandidate));

    if (!new_data) {
        return MIN_PQ_ALLOC_ERROR;
    }

    memcpy(new_data, pq->data, pq->size * sizeof(struct MergeCandidate));
    pq->data = new_data;
    pq->capacity = new_capacity;

    return MIN_PQ_SUCCESS;
}

static void heapify_up(struct MinPQ* pq, const size_t index) {
    if (index == 0) {
        return;
    }

    size_t parent_index = (index - 1) / 2;

    struct MergeCandidate* current = &pq->data[index];
    struct MergeCandidate* parent = &pq->data[parent_index];

    if (current->rank < parent->rank ||
        (current->rank == parent->rank &&
         current->left_idx < parent->left_idx)) {
        swap(current, parent);
        heapify_up(pq, parent_index);
    }
}

static void heapify_down(struct MinPQ* pq, const size_t index) {
    size_t left_child = 2 * index + 1;
    size_t right_child = 2 * index + 2;
    size_t smallest = index;

    if (left_child < pq->size) {
        struct MergeCandidate* smallest_cand = &pq->data[smallest];
        struct MergeCandidate* left_cand = &pq->data[left_child];
        if (left_cand->rank < smallest_cand->rank ||
            (left_cand->rank == smallest_cand->rank &&
             left_cand->left_idx < smallest_cand->left_idx)) {
            smallest = left_child;
        }
    }

    if (right_child < pq->size) {
        struct MergeCandidate* smallest_cand = &pq->data[smallest];
        struct MergeCandidate* right_cand = &pq->data[right_child];
        if (right_cand->rank < smallest_cand->rank ||
            (right_cand->rank == smallest_cand->rank &&
             right_cand->left_idx < smallest_cand->left_idx)) {
            smallest = right_child;
        }
    }

    if (smallest != index) {
        swap(&pq->data[index], &pq->data[smallest]);
        heapify_down(pq, smallest);
    }
}

static void swap(struct MergeCandidate* a, struct MergeCandidate* b) {
    struct MergeCandidate temp = *a;
    *a = *b;
    *b = temp;
}
