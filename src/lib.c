#include "hutoken/lib.h"

#include "Python.h"
#ifdef USE_FOMA
#include "fomalib.h"
#endif

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/bbpe.h"
#include "hutoken/bpe.h"
#include "hutoken/core.h"
#include "hutoken/hashmap.h"
#include "hutoken/helper.h"
#include "hutoken/string.h"
#include "hutoken/taskqueue.h"
#include "modsupport.h"
#include "object.h"
#include "pyerrors.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
typedef HANDLE thread_t;
typedef DWORD WINAPI thread_return_t;
typedef LPVOID thread_arg_t;
#define THREAD_CREATE(thr, func, arg) \
    *(thr) = CreateThread(NULL, 0, func, arg, 0, NULL)
#define THREAD_JOIN(thr) WaitForSingleObject(thr, INFINITE)

#else
#include <pthread.h>
typedef pthread_t thread_t;
typedef void* thread_return_t;
typedef void* thread_arg_t;
#define THREAD_CREATE(thr, func, arg) pthread_create(thr, NULL, func, arg)
#define THREAD_JOIN(thr) pthread_join(thr, NULL)
#endif

thread_return_t encode_wrapper(thread_arg_t arg) {
    TaskQueue* q = (TaskQueue*)arg;
    struct EncodeTask* task = NULL;

    while ((task = taskqueue_get(q)) != NULL) {
        encode(task);
    }

    return 0;
}

thread_return_t decode_wrapper(thread_arg_t arg) {
    DecodeQueue* q = (DecodeQueue*)arg;
    struct DecodeTask* task = NULL;

    while ((task = decodequeue_get(q)) != NULL) {
        decode(task);
    }

    return 0;
}

static char* pattern =
    "[ ]?[A-Za-záéíóúőűüöÁÉÍÓÚŐÜŰÖ]+|[ ]?[0-9]+|[ "
    "]?[^[:space:][:alpha:][:digit:]]+|[ ]+";
#define MAX_LINE_LENGTH 10000

struct EncodeContext* global_encode_context;
struct DecodeContext* global_decode_context;

PyObject* p_bpe_train(PyObject* self, PyObject* args) {
    char* data = NULL;
    char* vocab_file_name = NULL;
    int vocab_size = 256;

    if (!PyArg_ParseTuple(args, "sis", &data, &vocab_size, &vocab_file_name)) {
        return NULL;
    }

    if (vocab_size < 256) {
        PyErr_SetString(PyExc_RuntimeError,
                        "vocab_size must be at least 256 to encode all bytes.");
        return NULL;
    }
    size_t len = strlen(vocab_file_name);
    if (len < 4 || strcmp(vocab_file_name + (len - 4), ".txt") != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "vocab_file_name file extension must be .txt.");
        return NULL;
    }

    bpe_train(data, vocab_size, pattern, vocab_file_name);

    return Py_None;
}

PyObject* p_bbpe_train(PyObject* self, PyObject* args) {
    char* data = NULL;
    char* vocab_file_name = NULL;
    int vocab_size = 256;

    if (!PyArg_ParseTuple(args, "sis", &data, &vocab_size, &vocab_file_name)) {
        return NULL;
    }

    if (vocab_size < 256) {
        PyErr_SetString(PyExc_RuntimeError,
                        "vocab_size must be at least 256 to encode all bytes.");
        return NULL;
    }
    size_t len = strlen(vocab_file_name);
    if (len < 4 || strcmp(vocab_file_name + (len - 4), ".txt") != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "vocab_file_name file extension must be .txt.");
        return NULL;
    }

    bbpe_train(data, vocab_size, vocab_file_name);

    return Py_None;
}

int initialize_context(void) {
    global_encode_context = malloc(sizeof(struct EncodeContext));
    if (!global_encode_context) {
        log_debug("Error: Failed to allocate memory for encode_context.");
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to allocate memory for encode_context.");
        return -1;
    }
    memset(global_encode_context, 0, sizeof(struct EncodeContext));

    global_decode_context = malloc(sizeof(struct DecodeContext));
    if (!global_decode_context) {
        log_debug("Error: Failed to allocate memory for decode_context.");
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to allocate memory for decode_context.");
        return -1;
    }
    memset(global_decode_context, 0, sizeof(struct DecodeContext));

    global_encode_context->prefix = NULL;
    global_encode_context->is_byte_encoder = false;
    global_encode_context->initialized_encode = false;
    global_encode_context->pattern = pattern;

    global_encode_context->vocab_encode = hashmap_new(256);
    if (!global_encode_context->vocab_encode) {
        log_debug("Error: Failed to create hashmap for vocab_encode.");
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to create hashmap for vocab_encode.");
        return -1;
    }

    global_decode_context->vocab_size_decode = 0;
    global_decode_context->prefix = NULL;
    global_decode_context->is_byte_encoder = false;
    global_decode_context->initialized_decode = false;

    return 1;
}

static PyObject* p_initialize(PyObject* self,
                              PyObject* args,
                              PyObject* kwargs) {
    static char* kwlist[] = {
        "vocab_file_path",  "special_file_path", "prefix", "is_byte_encoder",
        "special_token_id", "pattern",           NULL};
    char* vocab_file_path = NULL;
    char* special_file_path = NULL;
    char* local_prefix = NULL;
    int local_is_byte_encoder = 0;
    int special_token_id = -1;  // Optional parameter for special token ID
    char* local_pattern = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sszp|iz", kwlist,
                                     &vocab_file_path, &special_file_path,
                                     &local_prefix, &local_is_byte_encoder,
                                     &special_token_id, &local_pattern)) {
        log_debug("Error: Invalid arguments passed to initialize.");
        PyErr_SetString(PyExc_TypeError,
                        "Invalid arguments. Expected a string "
                        "(vocab_file_path), a string (special_file_path), "
                        "a string or None (prefix) a bool an"
                        "optional integer (special_token_id) and"
                        " an optional string (regex_pattern)");
        return NULL;
    }

    log_debug("Initializing context for encode and decode");
    if (initialize_context() == -1) {
        return NULL;
    }

    if (local_prefix) {
        global_encode_context->prefix = strdup(local_prefix);
        global_decode_context->prefix = strdup(local_prefix);
    }

    global_encode_context->is_byte_encoder = local_is_byte_encoder;
    global_decode_context->is_byte_encoder = local_is_byte_encoder;

    if (local_pattern) {
        global_encode_context->pattern = strdup(local_pattern);
    }

    log_debug("Initializing with vocab file: %s", vocab_file_path);

    struct String hex_buffer;
    if (string_with_capacity(&hex_buffer, 1024) == STRING_ALLOC_ERROR) {
        hashmap_free(global_encode_context->vocab_encode);
        PyErr_NoMemory();
        return NULL;
    }

    FILE* file = fopen(vocab_file_path, "r");
    if (!file) {
        hashmap_free(global_encode_context->vocab_encode);
        string_release(&hex_buffer);
        log_debug("Error: Could not open vocab file: %s", vocab_file_path);
        PyErr_SetString(PyExc_FileNotFoundError, "Could not open vocab file.");
        return NULL;
    }

    log_debug("Sucessfully opened vocab file.");

    global_decode_context->vocab_size_decode = 0;
    char chunk[1024];

    struct String line;
    string_with_capacity(&line, 1024);
    while (true) {
        string_clear(&hex_buffer);
        string_clear(&line);
        bool at_eof = false;

        while (true) {
            if (!fgets(chunk, sizeof(chunk), file)) {
                at_eof = true;
                break;
            }

            if (string_append(&line, chunk) != STRING_SUCCESS) {
                log_debug("Error: Invalid format in vocab file: %s",
                          string_c_str(&line));
                (void)fclose(file);
                hashmap_free(global_encode_context->vocab_encode);
                string_release(&hex_buffer);
                string_release(&line);
                PyErr_SetString(PyExc_ValueError,
                                "Invalid format in vocab file.");
                return NULL;
            }

            if (strchr(chunk, '\n') != NULL) {
                break;
            }
        }

        if (at_eof) {
            break;
        }

        if (string_len(&line) == 0) {
            break;
        }

        char* separator = strstr(string_c_str(&line), " == ");
        if (separator == NULL) {
            log_debug("Error: Invalid format in vocab file: %s",
                      string_c_str(&line));
            (void)fclose(file);
            hashmap_free(global_encode_context->vocab_encode);
            string_release(&hex_buffer);
            string_release(&line);
            PyErr_SetString(PyExc_ValueError, "Invalid format in vocab file.");
            return NULL;
        }

        ptrdiff_t hex_len = separator - string_c_str(&line);

        string_append_n(&hex_buffer, string_c_str(&line), hex_len);

        char* value_str = separator + 4;  // strlen(" == ")
        char* endptr = NULL;
        errno = 0;

        long value = strtol(value_str, &endptr, 10);

        if (endptr == value_str) {
            (void)fclose(file);
            hashmap_free(global_encode_context->vocab_encode);
            string_release(&hex_buffer);
            log_debug("Error: No digits were found for value in line: '%s'.",
                      string_c_str(&line));
            string_release(&line);
            PyErr_SetString(
                PyExc_ValueError,
                "Invalid vocab format: could not parse integer value.");
            return NULL;
        }

        if (errno == ERANGE || value > INT_MAX || value < INT_MIN) {
            (void)fclose(file);
            hashmap_free(global_encode_context->vocab_encode);
            string_release(&hex_buffer);
            string_release(&line);
            log_debug("Error: Integer value '%s' is out of range.", value_str);
            PyErr_SetString(PyExc_ValueError,
                            "Integer value in vocab file is out of range.");
            return NULL;
        }

        char ascii_str[2048];
        hex_str_to_ascii(string_c_str(&hex_buffer), ascii_str,
                         sizeof(ascii_str));

        if (ascii_str[0] == '\0') {
            log_debug("Error: Failed to convert hex string to ASCII: %s",
                      string_c_str(&hex_buffer));
            (void)fclose(file);
            hashmap_free(global_encode_context->vocab_encode);
            string_release(&hex_buffer);
            string_release(&line);
            PyErr_SetString(PyExc_ValueError,
                            "Failed to convert hex string to ASCII.");
            return NULL;
        }

        char* durable_ascii_str = strdup(ascii_str);
        if (!durable_ascii_str) {
            log_debug("Error: Failed to convert hex string to ASCII: %s",
                      string_c_str(&hex_buffer));
            (void)fclose(file);
            hashmap_free(global_encode_context->vocab_encode);
            string_release(&hex_buffer);
            string_release(&line);
            PyErr_SetString(PyExc_ValueError,
                            "Failed to convert hex string to ASCII.");
            return NULL;
        }

        hashmap_set(
            global_encode_context->vocab_encode,
            &(struct Token){.key = durable_ascii_str, .value = (int)value});

        log_debug("Added vocab entry for encoding: key=%s, value=%d",
                  durable_ascii_str, value);

        global_decode_context->vocab_size_decode++;
    }

    if (global_decode_context->vocab_size_decode == 0) {
        (void)fclose(file);
        hashmap_free(global_encode_context->vocab_encode);
        string_release(&hex_buffer);
        string_release(&line);
        log_debug("Error: Vocab file is empty.");
        PyErr_SetString(PyExc_ValueError, "Vocab file is empty.");
        return NULL;
    }

    global_encode_context->initialized_encode = true;

    global_decode_context->vocab_decode = (char**)malloc(
        global_decode_context->vocab_size_decode * sizeof(char*));
    if (!global_decode_context->vocab_decode) {
        (void)fclose(file);
        hashmap_free(global_encode_context->vocab_encode);
        string_release(&hex_buffer);
        string_release(&line);
        log_debug("Error: Memory allocation failed for vocab_decode array.");
        PyErr_SetString(PyExc_MemoryError,
                        "Memory allocation failed for vocab_decode array.");
        return NULL;
    }

    size_t iter = 0;
    void* item = NULL;
    while (hashmap_iter(global_encode_context->vocab_encode, &iter, &item)) {
        const struct Token* token = item;

        global_decode_context->vocab_decode[token->value] = token->key;
        if (!global_decode_context->vocab_decode[token->value]) {
            (void)fclose(file);
            hashmap_free(global_encode_context->vocab_encode);
            string_release(&hex_buffer);
            string_release(&line);
            free((void*)global_decode_context->vocab_decode);
            log_debug(
                "Error: Memory allocation failed for vocab entry at index %d.",
                token->value);
            PyErr_SetString(PyExc_MemoryError,
                            "Memory allocation failed for vocab entry.");
            return NULL;
        }

        log_debug("Loaded vocab entry for decoding: index=%d, value=%s",
                  token->value, token->key);
    }

    log_debug("Successfully processed vocab file.");

    global_decode_context->initialized_decode = true;

    (void)fclose(file);
    string_release(&hex_buffer);
    string_release(&line);

    FILE* special_chars_file = fopen(special_file_path, "r");
    if (!special_chars_file) {
        hashmap_free(global_encode_context->vocab_encode);
        free((void*)global_decode_context->vocab_decode);
        log_debug("Error: Could not open special characters file: %s",
                  vocab_file_path);
        PyErr_SetString(PyExc_FileNotFoundError,
                        "Could not open special characters file.");
        return NULL;
    }

    log_debug("Successfully opened special character file.");

    char special_file_line[32];

    while (fgets(special_file_line, sizeof(special_file_line),
                 special_chars_file)) {
        char* separator = strstr(special_file_line, " == ");
        if (separator == NULL) {
            log_debug("Error: Invalid format in special character file: %s",
                      &special_file_line);
            (void)fclose(special_chars_file);
            hashmap_free(global_encode_context->vocab_encode);
            free((void*)global_decode_context->vocab_decode);
            PyErr_SetString(PyExc_ValueError,
                            "Invalid format in special character file.");
            return NULL;
        }

        char* endptr = NULL;
        errno = 0;

        long index = strtol(special_file_line, &endptr, 10);

        if (endptr == special_file_line) {
            log_debug("Error: No digits were found for value in line: '%s'.",
                      &special_file_line);
            (void)fclose(special_chars_file);
            hashmap_free(global_encode_context->vocab_encode);
            free((void*)global_decode_context->vocab_decode);
            PyErr_SetString(
                PyExc_ValueError,
                "Invalid vocab format: could not parse integer value.");
            return NULL;
        }

        if (errno == ERANGE || index > 256 || index < 0) {
            log_debug("Error: Integer value in line '%s' is out of range.",
                      special_file_line);
            (void)fclose(special_chars_file);
            hashmap_free(global_encode_context->vocab_encode);
            free((void*)global_decode_context->vocab_decode);
            PyErr_SetString(PyExc_ValueError,
                            "Integer value in vocab file is out of range.");
            return NULL;
        }

        char* value_str = separator + 4;  // strlen(" == "), again
        size_t value_len = strlen(value_str);
        char* value = malloc(value_len);
        memcpy(value, value_str, value_len - 1);
        value[value_len - 1] = '\0';

        if (value[0] == '\0') {
            log_debug("Error: Invalid replacement value in line '%s'.",
                      &special_file_line);
            free(value);
            (void)fclose(special_chars_file);
            hashmap_free(global_encode_context->vocab_encode);
            free((void*)global_decode_context->vocab_decode);
            PyErr_SetString(PyExc_ValueError,
                            "Failed to convert hex string to ASCII.");
            return NULL;
        }

        log_debug(
            "Loaded special character for pretokenization: key=%d, value='%s'",
            index, value);

        global_encode_context->special_chars[index] = strdup(value);
        ;
        global_decode_context->special_chars[index] = strdup(value);
    }

    (void)fclose(special_chars_file);

    Py_RETURN_NONE;
}

PyObject* p_encode(PyObject* self, PyObject* args) {
    struct EncodeContext* ctx = global_encode_context;

    if (!ctx || !ctx->initialized_encode) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Vocabulary is not initialized for encoding. "
                        "Call 'initialize_encode' function first.");
        return NULL;
    }

    char* text = NULL;

    if (!PyArg_ParseTuple(args, "s", &text)) {
        return NULL;
    }

    if (!text) {
        PyErr_SetString(PyExc_ValueError, "Text is empty.");
        return NULL;
    }

    int tokens_size = 0;
    int tokens[strlen(text)];

    encode(&(struct EncodeTask){
        .text = text,
        .ctx = ctx,
        .tokens = tokens,
        .tokens_size = &tokens_size,
        .error_msg = NULL,
    });

    PyObject* list = PyList_New(tokens_size);
    if (!list) {
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < tokens_size; i++) {
        PyObject* item = PyLong_FromLong(tokens[i]);
        if (!item) {
            Py_DECREF(list);  // cleanup in case of error
            PyErr_NoMemory();
            return NULL;
        }
        PyList_SetItem(list, i, item);
    }
    return list;
}

PyObject* p_batch_encode(PyObject* self, PyObject* args) {
    struct EncodeContext* ctx = global_encode_context;
    thread_t* threads = NULL;
    struct EncodeTask* tasks = NULL;
    PyObject* texts = NULL;
    int num_threads = 1;
    int num_texts = 0;

    if (!ctx || !ctx->initialized_encode) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Vocabulary is not initialized for encoding. "
                        "Call 'initialize_encode' function first.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O|i", &texts, &num_threads)) {
        log_debug("Error: Invalid arguments passed to encode.");
        PyErr_SetString(PyExc_TypeError,
                        "Invalid arguments. Expected a list of strings.");
        return NULL;
    }

    if (!PyList_Check(texts)) {
        log_debug("Error: Expected a list of strings.");
        PyErr_SetString(PyExc_TypeError,
                        "Invalid arguments. Expected a list of strings.");
        return NULL;
    }

    threads = malloc(num_threads * sizeof(thread_t));
    num_texts = PyList_Size(texts);
    tasks = malloc(num_texts * sizeof(struct EncodeTask));

    for (Py_ssize_t i = 0; i < num_texts; i++) {
        PyObject* item = PyList_GetItem(texts, i);
        if (!item) {
            log_debug("Error: Failed to get item at index %zd", i);
            PyErr_SetString(PyExc_RuntimeError, "Failed to get item.");
            return NULL;
        }

        char* text_chunk = (char*)PyUnicode_AsUTF8(item);

        tasks[i].text = strdup(text_chunk);
        tasks[i].ctx = ctx;
        tasks[i].tokens = malloc(sizeof(int) * strlen(text_chunk) * 4 + 10);
        tasks[i].tokens_size = malloc(sizeof(int));
        *tasks[i].tokens_size = 0;
        tasks[i].error_msg = NULL;
    }

    TaskQueue q;
    taskqueue_init(&q, tasks, num_texts);

    Py_BEGIN_ALLOW_THREADS

        for (int i = 0; i < num_threads; i++) {
        log_debug("Starting thread number %d", i);
        THREAD_CREATE(&threads[i], encode_wrapper, &q);
    }

    for (int i = 0; i < num_threads; i++) {
        THREAD_JOIN(threads[i]);
    }
    log_debug("All threads joined");

    Py_END_ALLOW_THREADS

        for (Py_ssize_t i = 0; i < num_texts; i++) {
        if (tasks[i].error_msg) {
            log_debug("Error occurred in chunk %zd: %s", i, tasks[i].error_msg);
            PyErr_SetString(PyExc_RuntimeError, tasks[i].error_msg);
            return NULL;
        }
    }

    PyObject* result = PyList_New(num_texts);
    if (!result) {
        log_debug("Error: Failed to create result list");
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_texts; i++) {
        int size = *tasks[i].tokens_size;

        PyObject* sublist = PyList_New(size);
        if (!sublist) {
            Py_DECREF(result);
            log_debug("Error: Failed to create sublist for chunk %zd", i);
            PyErr_NoMemory();
            return NULL;
        }

        log_debug("Inserting tokens for chunk %zd, size: %d", i, size);

        for (int j = 0; j < size; j++) {
            PyObject* item = PyLong_FromLong(tasks[i].tokens[j]);
            if (!item) {
                Py_DECREF(sublist);
                Py_DECREF(result);
                PyErr_NoMemory();
                return NULL;
            }
            PyList_SetItem(sublist, j, item);
        }

        PyList_SetItem(result, i, sublist);
    }

    for (Py_ssize_t i = 0; i < num_texts; i++) {
        free(tasks[i].text);
        free(tasks[i].tokens);
        free(tasks[i].tokens_size);
    }

    free(threads);
    free(tasks);

    return result;
}

static PyObject* p_decode(PyObject* self, PyObject* args) {
    struct DecodeContext* ctx = global_decode_context;
    PyObject* tokens = NULL;
    int tokens_size = 0;
    struct DecodeTask* task = malloc(sizeof(struct DecodeTask));

    if (!ctx || !ctx->initialized_decode) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Vocabulary is not initialized for decoding. "
                        "Call 'initialize_decode' function first.");
        return NULL;
    }

    if (!ctx->vocab_size_decode) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Vocab size is not properly set during initialization. "
                        "Please try again.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O", &tokens)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Failed to parse arguments. Expected a single list of tokens.");
        return NULL;
    }

    if (!PyList_Check(tokens)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of integers");
        return NULL;
    }

    tokens_size = PyList_Size(tokens);
    char* result = NULL;
    char* error_msg = NULL;
    int* token_array = malloc(sizeof(int) * tokens_size);

    for (int i = 0; i < tokens_size; i++) {
        PyObject* item = PyList_GetItem(tokens, i);
        if (!item) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to get token item.");
            return NULL;
        }

        token_array[i] = (int)PyLong_AsLong(item);
    }

    task->tokens = token_array;
    task->tokens_size = &tokens_size;
    task->result = result;
    task->ctx = ctx;
    task->error_msg = error_msg;

    decode(task);

    if (task->error_msg) {
        PyErr_SetString(PyExc_ValueError, task->error_msg);
        free(task->tokens);
        free(task);
        return NULL;
    }

    return task->result ? PyUnicode_FromString(task->result) : Py_None;
}

static PyObject* p_batch_decode(PyObject* self, PyObject* args) {
    struct DecodeContext* ctx = global_decode_context;
    thread_t* threads = NULL;
    struct DecodeTask* tasks = NULL;
    PyObject* tokens = NULL;
    Py_ssize_t num_tokens = 0;
    int num_threads = 1;

    if (!ctx || !ctx->initialized_decode) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Vocabulary is not initialized for decoding. "
                        "Call 'initialize_decode' function first.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O|i", &tokens, &num_threads)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Failed to parse arguments. Expected a single list of tokens.");
        return NULL;
    }

    num_tokens = PyList_Size(tokens);
    if (num_tokens <= 0) {
        PyErr_SetString(PyExc_ValueError, "No tokens provided.");
        return NULL;
    }

    threads = malloc(num_threads * sizeof(thread_t));
    tasks = malloc(num_tokens * sizeof(struct DecodeTask));

    for (Py_ssize_t i = 0; i < num_tokens; i++) {
        PyObject* item = PyList_GetItem(tokens, i);
        if (!item) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to get token item.");
            return NULL;
        }

        if (!PyList_Check(item)) {
            log_debug("Error: item at index %zd is not a list", i);
            PyErr_SetString(PyExc_TypeError,
                            "Each item must be a list of integers.");
            return NULL;
        }

        int tokens_size = PyList_Size(item);
        tasks[i].tokens = malloc(sizeof(int) * tokens_size);
        if (!tasks[i].tokens) {
            log_debug("Error: Memory allocation failed for tokens");
            PyErr_SetString(PyExc_MemoryError,
                            "Failed to allocate memory for tokens");
            return NULL;
        }
        for (Py_ssize_t j = 0; j < tokens_size; j++) {
            PyObject* token = PyList_GetItem(item, j);
            if (token) {
                tasks[i].tokens[j] = (int)PyLong_AsLong(token);
            } else {
                tasks[i].tokens[j] = -1;
            }
        }

        tasks[i].tokens_size = malloc(sizeof(int));
        if (!tasks[i].tokens_size) {
            log_debug("Error: Memory allocation failed for tokens_size");
            PyErr_SetString(PyExc_MemoryError,
                            "Failed to allocate memory for tokens_size");
            return NULL;
        }
        *tasks[i].tokens_size = PyList_Size(item);
        tasks[i].result = NULL;
        tasks[i].ctx = ctx;
    }

    DecodeQueue q;
    decodequeue_init(&q, tasks, num_tokens);

    Py_BEGIN_ALLOW_THREADS

        for (int i = 0; i < num_threads; i++) {
        log_debug("Starting thread %d", i);
        THREAD_CREATE(&threads[i], decode_wrapper, &q);
    }

    for (int i = 0; i < num_threads; i++) {
        THREAD_JOIN(threads[i]);
    }
    log_debug("All threads joined");

    Py_END_ALLOW_THREADS

        for (Py_ssize_t i = 0; i < num_tokens; i++) {
        if (tasks[i].error_msg) {
            log_debug("Error occurred in chunk %zd: %s", i, tasks[i].error_msg);
            PyErr_SetString(PyExc_ValueError, tasks[i].error_msg);
            return NULL;
        }
    }

    PyObject* results_list = PyList_New(num_tokens);
    if (!results_list) {
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to allocate memory for result list");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_tokens; i++) {
        PyObject* string = PyUnicode_FromString(tasks[i].result);
        if (!string) {
            Py_DECREF(results_list);
            PyErr_SetString(PyExc_MemoryError,
                            "Failed to create Python string from decoded text");
            return NULL;
        }
        PyList_SET_ITEM(results_list, i, string);

        free(tasks[i].tokens_size);
        free(tasks[i].result);
        free(tasks[i].tokens);
    }

    return results_list;
}

#ifdef USE_FOMA
PyObject* p_initialize_foma(PyObject* self) {
    return initialize_foma();
}

PyObject* p_look_up_word(PyObject* self, PyObject* args) {
    PyObject* py_handle = NULL;
    struct apply_handle* handle = NULL;
    char* word = NULL;
    bool only_longest = false;

    if (!PyArg_ParseTuple(args, "Os|b", &py_handle, &word, &only_longest)) {
        PyErr_SetString(PyExc_TypeError,
                        "Function takes three arguments: (apply_handle, word, "
                        "only_longest).");
        return NULL;
    }

    handle = (struct apply_handle*)PyCapsule_GetPointer(py_handle,
                                                        "foma.apply_handle");

    if (handle == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "Argument must be an apply_handle struct, returned by "
                        "`hutoken.initialize_foma()`.");
        return NULL;
    }

    return look_up_word(handle, word, only_longest);
}
#endif

static PyMethodDef huTokenMethods[] = {
    {"bpe_train", p_bpe_train, METH_VARARGS, "BPE training"},
    {"bbpe_train", p_bbpe_train, METH_VARARGS, "BBPE training"},
    {"initialize", (PyCFunction)p_initialize, METH_VARARGS | METH_KEYWORDS,
     "Initalize tokenizer"},
    {"encode", (PyCFunction)p_encode, METH_VARARGS, "Encodes string"},
    {"batch_encode", (PyCFunction)p_batch_encode, METH_VARARGS,
     "Encodes list of strings"},
    {"decode", p_decode, METH_VARARGS, "Decodes list of ints"},
    {"batch_decode", p_batch_decode, METH_VARARGS,
     "Decodes list of lists of ints"},
#ifdef USE_FOMA
    {"initialize_foma", (PyCFunction)p_initialize_foma, METH_NOARGS,
     "Initilaizes the foma fst"},
    {"look_up_word", (PyCFunction)p_look_up_word, METH_VARARGS,
     "Morphological analysis of a word"},
#endif
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef huToken = {PyModuleDef_HEAD_INIT, "huToken",
                                     "hutoken module description", -1,
                                     huTokenMethods};

PyMODINIT_FUNC PyInit__hutoken(void) {
    return PyModule_Create(&huToken);
}
