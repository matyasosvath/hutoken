#include <stdbool.h>

#include "hutoken/taskqueue.h"

void taskqueue_init(TaskQueue* q, struct EncodeTask* tasks, int num_tasks) {
    q->tasks = tasks;
    q->num_tasks = num_tasks;
    q->next_task = 0;
#if defined(_WIN32) || defined(_WIN64)
    InitializeCriticalSection(&q->lock);
#else
    pthread_mutex_init(&q->lock, NULL);
#endif
}

struct EncodeTask* taskqueue_get(TaskQueue* q) {
    struct EncodeTask* t = NULL;
#if defined(_WIN32) || defined(_WIN64)
    EnterCriticalSection(&q->lock);
#else
    pthread_mutex_lock(&q->lock);
#endif

    if (q->next_task < q->num_tasks) {
        t = &q->tasks[q->next_task++];
    }

#if defined(_WIN32) || defined(_WIN64)
    LeaveCriticalSection(&q->lock);
#else
    pthread_mutex_unlock(&q->lock);
#endif

    return t;
}

void decodequeue_init(DecodeQueue *q, struct DecodeTask *tasks, int num_tasks){
    q->tasks = tasks;
    q->num_tasks = num_tasks;
    q->next_task = 0;
#if defined(_WIN32) || defined(_WIN64)
    InitializeCriticalSection(&q->lock);
#else
    pthread_mutex_init(&q->lock, NULL);
#endif
}

struct DecodeTask* decodequeue_get(DecodeQueue* q) {
    struct DecodeTask* t = NULL;
#if defined(_WIN32) || defined(_WIN64)
    EnterCriticalSection(&q->lock);
#else
    pthread_mutex_lock(&q->lock);
#endif

    if (q->next_task < q->num_tasks) {
        t = &q->tasks[q->next_task++];
    }

#if defined(_WIN32) || defined(_WIN64)
    LeaveCriticalSection(&q->lock);
#else
    pthread_mutex_unlock(&q->lock);
#endif

    return t;
}
