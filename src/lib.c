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
#include "modsupport.h"
#include "object.h"
#include "pyerrors.h"

static bool initialized_encode = false;
static bool initialized_decode = false;
static char* pattern =
    "[ ]?[A-Za-záéíóúőűüöÁÉÍÓÚŐÜŰÖ]+|[ ]?[0-9]+|[ "
    "]?[^[:space:][:alpha:][:digit:]]+|[ ]+";

struct HashMap* vocab_encode;
char** vocab_decode;
int vocab_size_decode;
char* special_chars[256];
char* prefix;
bool is_byte_encoder;
#define MAX_LINE_LENGTH 10000

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

PyObject* p_initialize(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {
        "vocab_file_path",  "special_file_path", "prefix", "is_byte_encoder",
        "special_token_id", "pattern",           NULL};
    char* vocab_file_path = NULL;
    char* special_file_path = NULL;
    char* local_prefix = NULL;
    int local_is_byte_encoder = 0;
    int special_token_id = -1;  // Optional parameter for special token ID
    prefix = NULL;
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

    if (local_prefix) {
        prefix = strdup(local_prefix);
    }
    is_byte_encoder = local_is_byte_encoder;

    if (local_pattern) {
        pattern = strdup(local_pattern);
    }

    log_debug("Initializing with vocab file: %s", vocab_file_path);

    vocab_encode = hashmap_new(256);
    if (!vocab_encode) {
        log_debug("Error: Failed to create hashmap for vocab_encode.");
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to create hashmap for vocab_encode.");
        return NULL;
    }

    struct String hex_buffer;
    if (string_with_capacity(&hex_buffer, 1024) == STRING_ALLOC_ERROR) {
        hashmap_free(vocab_encode);
        PyErr_NoMemory();
        return NULL;
    }

    FILE* file = fopen(vocab_file_path, "r");
    if (!file) {
        hashmap_free(vocab_encode);
        string_release(&hex_buffer);
        log_debug("Error: Could not open vocab file: %s", vocab_file_path);
        PyErr_SetString(PyExc_FileNotFoundError, "Could not open vocab file.");
        return NULL;
    }

    log_debug("Sucessfully opened vocab file.");

    vocab_size_decode = 0;
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
                hashmap_free(vocab_encode);
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
            hashmap_free(vocab_encode);
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
            hashmap_free(vocab_encode);
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
            hashmap_free(vocab_encode);
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
            hashmap_free(vocab_encode);
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
            hashmap_free(vocab_encode);
            string_release(&hex_buffer);
            string_release(&line);
            PyErr_SetString(PyExc_ValueError,
                            "Failed to convert hex string to ASCII.");
            return NULL;
        }

        hashmap_set(vocab_encode, &(struct Token){.key = durable_ascii_str,
                                                  .value = (int)value});

        log_debug("Added vocab entry for encoding: key=%s, value=%d",
                  durable_ascii_str, value);

        vocab_size_decode++;
    }

    if (vocab_size_decode == 0) {
        (void)fclose(file);
        hashmap_free(vocab_encode);
        string_release(&hex_buffer);
        string_release(&line);
        log_debug("Error: Vocab file is empty.");
        PyErr_SetString(PyExc_ValueError, "Vocab file is empty.");
        return NULL;
    }

    initialized_encode = true;

    vocab_decode = (char**)malloc(vocab_size_decode * sizeof(char*));
    if (!vocab_decode) {
        (void)fclose(file);
        hashmap_free(vocab_encode);
        string_release(&hex_buffer);
        string_release(&line);
        log_debug("Error: Memory allocation failed for vocab_decode array.");
        PyErr_SetString(PyExc_MemoryError,
                        "Memory allocation failed for vocab_decode array.");
        return NULL;
    }

    size_t iter = 0;
    void* item = NULL;
    while (hashmap_iter(vocab_encode, &iter, &item)) {
        const struct Token* token = item;

        vocab_decode[token->value] = token->key;
        if (!vocab_decode[token->value]) {
            (void)fclose(file);
            hashmap_free(vocab_encode);
            string_release(&hex_buffer);
            string_release(&line);
            free((void*)vocab_decode);
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

    initialized_decode = true;

    (void)fclose(file);
    string_release(&hex_buffer);
    string_release(&line);

    FILE* special_chars_file = fopen(special_file_path, "r");
    if (!special_chars_file) {
        hashmap_free(vocab_encode);
        free((void*)vocab_decode);
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
            hashmap_free(vocab_encode);
            free((void*)vocab_decode);
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
            hashmap_free(vocab_encode);
            free((void*)vocab_decode);
            PyErr_SetString(
                PyExc_ValueError,
                "Invalid vocab format: could not parse integer value.");
            return NULL;
        }

        if (errno == ERANGE || index > 256 || index < 0) {
            log_debug("Error: Integer value in line '%s' is out of range.",
                      special_file_line);
            (void)fclose(special_chars_file);
            hashmap_free(vocab_encode);
            free((void*)vocab_decode);
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
            hashmap_free(vocab_encode);
            free((void*)vocab_decode);
            PyErr_SetString(PyExc_ValueError,
                            "Failed to convert hex string to ASCII.");
            return NULL;
        }

        log_debug(
            "Loaded special character for pretokenization: key=%d, value='%s'",
            index, value);

        special_chars[index] = value;
    }

    (void)fclose(special_chars_file);

    Py_RETURN_NONE;
}

PyObject* p_encode(PyObject* self, PyObject* args) {
    if (!initialized_encode) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Vocabulary is not initialized for encoding. "
                        "Call 'initialize_encode' function first.");
        return NULL;
    }

    char* text = NULL;

    if (!PyArg_ParseTuple(args, "s", &text)) {
        return NULL;
    }

    int tokens_size = 0;
    int tokens[strlen(text)];

    encode(text, vocab_encode, pattern, tokens, &tokens_size,
           (const char**)special_chars, prefix, is_byte_encoder);

    PyObject* list = PyList_New(tokens_size);
    if (!list) {
        return PyErr_NoMemory();
    }

    for (int i = 0; i < tokens_size; i++) {
        PyObject* item = PyLong_FromLong(tokens[i]);
        if (!item) {
            Py_DECREF(list);  // cleanup in case of error
            return PyErr_NoMemory();
        }
        PyList_SetItem(list, i, item);
    }
    return list;
}

static PyObject* p_decode(PyObject* self, PyObject* args) {
    if (!initialized_decode) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Vocabulary is not initialized for decoding. "
                        "Call 'initialize_decode' function first.");
        return NULL;
    }

    if (!vocab_size_decode) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Vocab size is not properly set during initialization. "
                        "Please try again.");
        return NULL;
    }

    PyObject* tokens = NULL;

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

    return decode(tokens, vocab_decode, vocab_size_decode,
                  (const char**)special_chars, prefix, is_byte_encoder);
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
    {"encode", p_encode, METH_VARARGS, "Encodes string"},
    {"decode", p_decode, METH_VARARGS, "Decodes list of ints"},
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
