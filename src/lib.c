#include <Python.h>

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>

#include "helper.c"
#include "core.c"
#include "bpe.c"


static bool initialized_encode = false;
static bool initialized_decode = false;
static char *pattern = " ?[A-Za-záéíóúőüöÁÉÍÓÚŐÜÖ]+| ?[0-9]+| ?[^A-Za-z0-9\\s]+|\\s+";
static regex_t precompiled_regex;

struct HashMap *vocab_encode;
char **vocab_decode;
int vocab_size_decode;
#define MAX_LINE_LENGTH 10000 



PyObject *p_bpe_train(PyObject *self, PyObject *args)
{

    char *data;
    char *vocab_file_name;
    int vocab_size = 256;

    if (!PyArg_ParseTuple(args, "sis", &data, &vocab_size, &vocab_file_name))
        return NULL;

    if (vocab_size < 256) {
        PyErr_SetString(PyExc_RuntimeError,"vocab_size must be at least 256 to encode all bytes.");
        return NULL;
    }
    int len = strlen(vocab_file_name);
    if (len < 4 || strcmp(vocab_file_name + (len - 4), ".txt") != 0) {
        PyErr_SetString(PyExc_RuntimeError, "vocab_file_name file extension must be .txt.");
        return NULL;
    }

    bpe_train(data, vocab_size, pattern, vocab_file_name);

    return Py_None;
}

static PyObject *p_initialize(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"vocab_file_path", "special_token_id", NULL};
    char *vocab_file_path;
    int special_token_id = -1; // Optional parameter for special token ID

    // Parse arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|i", kwlist, &vocab_file_path, &special_token_id)) {
        log_debug("Error: Invalid arguments passed to initialize.");
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected a string (vocab_file_path) and an optional integer (special_token_id).");
        return NULL;
    }

    log_debug("Initializing with vocab file: %s", vocab_file_path);

    // Compile regex pattern once
    int r = regcomp(&precompiled_regex, pattern, REG_EXTENDED);
    if (r) {
        log_debug("Error: Regex could not be compiled.");
        PyErr_SetString(PyExc_RuntimeError, "Regex could not be compiled.");
        return NULL;
    }

    // Initialize encoding
    vocab_encode = hashmap_new(256);
    if (!vocab_encode) {
        log_debug("Error: Failed to create hashmap for vocab_encode.");
        PyErr_SetString(PyExc_MemoryError, "Failed to create hashmap for vocab_encode.");
        return NULL;
    }

    FILE *file = fopen(vocab_file_path, "r");
    if (!file) {
        log_debug("Error: Could not open vocab file: %s", vocab_file_path);
        PyErr_SetString(PyExc_FileNotFoundError, "Could not open vocab file.");
        return NULL;
    }

    log_debug("Successfully opened vocab file for encoding: %s", vocab_file_path);

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        char *hex_token = strtok(line, " == ");
        char *value_str = strtok(NULL, " == ");
        if (!hex_token || !value_str) {
            log_debug("Error: Invalid line format in vocab file: %s", line);
            continue;
        }

        size_t hex_len = strlen(hex_token);
        size_t decoded_string_len = hex_len / 4 + 1;
        char *decoded_string = malloc(decoded_string_len);
        if (!decoded_string) {
            log_debug("Error: Memory allocation failed for decoded string.");
            fclose(file);
            return PyErr_NoMemory();
        }

        const char *pos = hex_token;
        size_t char_index = 0;
        while (pos[0] == '0' && pos[1] == 'x') {
            unsigned int byte_value;
            if (sscanf(pos, "0x%2X", &byte_value) != 1) {
                log_debug("Error: Failed to parse hex byte: %s", pos);
                free(decoded_string);
                continue;
            }
            decoded_string[char_index++] = (char)byte_value;
            pos += 4;
        }
        decoded_string[char_index] = '\0';

        char *key = strdup(decoded_string);
        if (!key) {
            log_debug("Error: Memory allocation failed for key string.");
            free(decoded_string);
            fclose(file);
            return PyErr_NoMemory();
        }
        int value = atoi(value_str);

        hashmap_set(vocab_encode, &(struct Token){.key = key, .value = value});
        log_debug("Added vocab entry for encoding: key=%s, value=%d", key, value);

        free(decoded_string);
    }

    fclose(file);
    initialized_encode = true;
    log_debug("Successfully initialized encoding.");

    // Initialize decoding
    file = fopen(vocab_file_path, "r");
    if (!file) {
        log_debug("Error: Could not open vocab file for decoding: %s", vocab_file_path);
        PyErr_SetString(PyExc_FileNotFoundError, "Could not open vocab file.");
        return NULL;
    }

    log_debug("Successfully opened vocab file for decoding: %s", vocab_file_path);

    // Count the number of lines in the file to determine vocab size
    vocab_size_decode = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strchr(line, '=') != NULL) { // Ensure the line contains a valid entry
            vocab_size_decode++;
        }
    }

    log_debug("Calculated vocab size for decoding: %d", vocab_size_decode);

    if (vocab_size_decode == 0) {
        log_debug("Error: Vocab file is empty or contains no valid entries.");
        PyErr_SetString(PyExc_ValueError, "Vocab file is empty or contains no valid entries.");
        fclose(file);
        return NULL;
    }

    vocab_decode = malloc(vocab_size_decode * sizeof(char *));
    if (!vocab_decode) {
        log_debug("Error: Memory allocation failed for vocab_decode array.");
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for vocab_decode array.");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < vocab_size_decode; i++) {
        vocab_decode[i] = NULL;
    }

    rewind(file);

    char *hex_str = malloc(1024);
    if (!hex_str) {
        log_debug("Error: Memory allocation failed for hex_str.");
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for hex_str.");
        fclose(file);
        free(vocab_decode);
        return NULL;
    }

    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        int value;

        if (sscanf(line, "%9999s == %d", hex_str, &value) != 2) {
            log_debug("Error: Invalid format in vocab file: %s", line);
            PyErr_SetString(PyExc_ValueError, "Invalid format in vocab file.");
            fclose(file);
            free(hex_str);
            free(vocab_decode);
            return NULL;
        }

        char ascii_str[500] = {0};
        hex_str_to_ascii(hex_str, ascii_str, sizeof(ascii_str));

        if (ascii_str[0] == '\0') {
            log_debug("Error: Failed to convert hex string to ASCII: %s", hex_str);
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert hex string to ASCII.");
            fclose(file);
            free(hex_str);
            free(vocab_decode);
            return NULL;
        }

        vocab_decode[value] = strdup(ascii_str);
        if (!vocab_decode[value]) {
            log_debug("Error: Memory allocation failed for vocab entry at index %d.", value);
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for vocab entry.");
            fclose(file);
            free(hex_str);
            free(vocab_decode);
            return NULL;
        }

        log_debug("Loaded vocab entry for decoding: index=%d, value=%s", value, ascii_str);

        index++;
    }

    free(hex_str);
    fclose(file);

    initialized_decode = true;
    log_debug("Successfully initialized decoding.");

    Py_RETURN_NONE;
}

PyObject *p_encode(PyObject *self, PyObject *args)
{

    if (!initialized_encode)
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Vocabulary is not initialized for encoding. "
            "Call 'initialize_encode' function first."
        );
        return NULL;
    }

    char *text;
    if (!PyArg_ParseTuple(args, "s", &text))
        return NULL;

    int tokens_size = 0;
    int tokens[strlen(text)];

    encode(text, vocab_encode, &precompiled_regex, tokens, &tokens_size);

    PyObject *list = PyList_New(tokens_size);
    if (!list)
        return PyErr_NoMemory();

    for (int i = 0; i < tokens_size; i++)
    {
        PyObject *item = PyLong_FromLong(tokens[i]);
        if (!item)
        {
            Py_DECREF(list); // cleanup in case of error
            return PyErr_NoMemory();
        }
        PyList_SetItem(list, i, item);
    }
    return list;
}

static PyObject *p_decode(PyObject *self, PyObject *args) {
    if (!initialized_decode) {
        PyErr_SetString(PyExc_RuntimeError,
            "Vocabulary is not initialized for decoding. "
            "Call 'initialize_decode' function first."
        );
        return NULL;
    }

    if (!vocab_size_decode) {
        PyErr_SetString(PyExc_RuntimeError,
            "Vocab size is not properly set during initialization. "
            "Please try again."
        );
        return NULL;
    }

    PyObject *tokens;

    if (!PyArg_ParseTuple(args, "O", &tokens)) {
        PyErr_SetString(PyExc_TypeError, "Failed to parse arguments. Expected a single list of tokens.");
        return NULL;
    }

    if (!PyList_Check(tokens)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of integers");
        return NULL;
    }

    return decode(tokens, vocab_decode, vocab_size_decode);
}

static PyMethodDef huTokenMethods[] = {
    {"bpe_train", p_bpe_train, METH_VARARGS, "BPE training"},
    {"initialize", (PyCFunction)p_initialize, METH_VARARGS | METH_KEYWORDS, "Initalize tokenizer"},
    {"encode", p_encode, METH_VARARGS, "Encodes string"},
    {"decode", p_decode, METH_VARARGS, "Decodes list of ints"},
    {NULL, NULL, 0, NULL} 
};

static struct PyModuleDef huToken = {
    PyModuleDef_HEAD_INIT,
    "huToken",
    "hutoken module description",
    -1,
    huTokenMethods
};

PyMODINIT_FUNC PyInit_hutoken(void) {
    return PyModule_Create(&huToken);
}