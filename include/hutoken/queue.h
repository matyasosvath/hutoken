#ifndef HUTOKEN_QUEUE_H
#define HUTOKEN_QUEUE_H

#include <stdbool.h>
#include <stddef.h>

#include "hutoken/arena.h"

struct MergeCandidate {
    int rank;
    size_t left_idx;
    size_t right_idx;
};

struct MinPQ {
    struct MergeCandidate* data;
    size_t size;
    size_t capacity;
};

enum MinPQError {
    MIN_PQ_SUCCESS,
    MIN_PQ_INVALID_ARGUMENT,
    MIN_PQ_ALLOC_ERROR,
    MIN_PQ_EMPTY,
};

enum MinPQError min_pq_init(struct MinPQ* pq, const size_t capacity);
void min_pq_release(struct MinPQ* pq);
enum MinPQError min_pq_push(struct MinPQ* pq,
                            const struct MergeCandidate candidate);
enum MinPQError min_pq_pop(struct MinPQ* pq, struct MergeCandidate* candidate);
size_t min_pq_size(const struct MinPQ* pq);
bool min_pq_is_empty(const struct MinPQ* pq);

enum MinPQError min_pq_init_arena(struct Arena* arena,
                                  struct MinPQ* pq,
                                  const size_t capacity);
enum MinPQError min_pq_push_arena(struct Arena* arena,
                                  struct MinPQ* pq,
                                  const struct MergeCandidate candidate);

#endif
