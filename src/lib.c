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
static char *pattern = "[ ]?[A-Za-záéíóúőűüöÁÉÍÓÚŐÜŰÖ]+|[ ]?[0-9]+|[ ]?[^[:space:][:alpha:][:digit:]]+|[ ]+";
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
    int special_token_id = -1;

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

    // --- First pass: find max token value for vocab_decode size ---
    FILE *file = fopen(vocab_file_path, "r");
    if (!file) {
        log_debug("Error: Could not open vocab file: %s", vocab_file_path);
        PyErr_SetString(PyExc_FileNotFoundError, "Could not open vocab file.");
        return NULL;
    }

    int max_value = -1;
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        int value;
        if (sscanf(line, "%*s == %d", &value) == 1) {
            if (value > max_value) max_value = value;
        }
    }
    if (max_value < 0) {
        fclose(file);
        log_debug("Error: No valid entries in vocab file.");
        PyErr_SetString(PyExc_ValueError, "No valid entries in vocab file.");
        return NULL;
    }
    vocab_size_decode = max_value + 1;

    // Allocate decode array and encoding hashmap
    vocab_decode = calloc(vocab_size_decode, sizeof(char *));
    if (!vocab_decode) {
        fclose(file);
        log_debug("Error: Memory allocation failed for vocab_decode array.");
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for vocab_decode array.");
        return NULL;
    }
    vocab_encode = hashmap_new(256);
    if (!vocab_encode) {
        fclose(file);
        free(vocab_decode);
        log_debug("Error: Failed to create hashmap for vocab_encode.");
        PyErr_SetString(PyExc_MemoryError, "Failed to create hashmap for vocab_encode.");
        return NULL;
    }

    // --- Second pass: fill both encode and decode structures ---
    rewind(file);
    while (fgets(line, sizeof(line), file)) {
        char hex_token[512];
        int value;
        if (sscanf(line, "%s == %d", hex_token, &value) == 2) {
            char ascii_str[512] = {0};
            hex_str_to_ascii(hex_token, ascii_str, sizeof(ascii_str));
            char *key = strdup(ascii_str);
            if (key) {
                hashmap_set(vocab_encode, &(struct Token){.key = key, .value = value});
            }
            if (!vocab_decode[value]) {
                vocab_decode[value] = strdup(ascii_str);
            }
        }
    }
    fclose(file);

    initialized_encode = true;
    initialized_decode = true;
    log_debug("Successfully initialized encoding and decoding.");

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

PyMODINIT_FUNC PyInit__hutoken(void) {
    return PyModule_Create(&huToken);
}
