#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <stdbool.h>

typedef struct {
    struct EncodeTask* tasks;
    int num_tasks;
    int next_task;
    #if defined(_WIN32) || defined(_WIN64)
    CRITICAL_SECTION lock;
    #else
    pthread_mutex_t lock;
    #endif
} TaskQueue;

void taskqueue_init(TaskQueue* q, struct EncodeTask* tasks, int num_tasks);
struct EncodeTask* taskqueue_get(TaskQueue* q);
