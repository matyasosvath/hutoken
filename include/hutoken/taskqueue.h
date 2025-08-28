#ifndef HUTOKEN_TASKQUEUE_H
#define HUTOKEN_TASKQUEUE_H

#include "hutoken/ac.h"
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <stdbool.h>

#include "hutoken/hashmap.h"

struct EncodeContext {
    bool initialized_encode;
    struct HashMap* vocab_encode;
    struct MergeRule* merge_rules;
    size_t num_merge_rules;
    struct HashMap* merges_map;
    char* pattern;
    char* special_chars[256];
    char* prefix;
    bool is_byte_encoder;
};

struct DecodeContext {
    bool initialized_decode;
    char** vocab_decode;
    size_t* vocab_decode_lens;
    int vocab_size_decode;
    char* special_chars[256];
    char* prefix;
    bool is_byte_encoder;
    struct HashMap* special_chars_map_decode;
    size_t max_special_char_len;
    struct ACAutomaton* ac;
};

struct EncodeTask {
    char* text;
    struct EncodeContext* ctx;
    int* tokens;
    int* tokens_size;
    char* error_msg;
};

struct DecodeTask {
    int* tokens;
    int* tokens_size;
    struct DecodeContext* ctx;
    char* result;
    char* error_msg;
};

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

typedef struct {
    struct DecodeTask* tasks;
    int num_tasks;
    int next_task;
#if defined(_WIN32) || defined(_WIN64)
    CRITICAL_SECTION lock;
#else
    pthread_mutex_t lock;
#endif
} DecodeQueue;

void decodequeue_init(DecodeQueue* q, struct DecodeTask* tasks, int num_tasks);
struct DecodeTask* decodequeue_get(DecodeQueue* q);

#endif
