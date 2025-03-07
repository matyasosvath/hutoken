#include <Python.h>

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "helper.c"
#include "core.c"
#include "bpe.c"


static bool initialized_encode = false;
static bool initialized_decode = false;
static int buffer_size = 100;
static char *pattern = " ?[A-Za-záéíóúőüöÁÉÍÓÚŐÜÖ]+| ?[0-9]+| ?[^A-Za-z0-9\\s]+|\\s+";

struct HashMap *vocab_encode;
char **vocab_decode;
int vocab_size_decode = 0;



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

PyObject *p_initalize_encode(PyObject *self, PyObject *args)
{
    char *vocab_file_path; // full file path for now

    if (!PyArg_ParseTuple(args, "s", &vocab_file_path))
        return NULL;

    vocab_encode = hashmap_new(256);

    FILE *file = fopen(vocab_file_path, "r");
    if (!file)
    {
        perror("Could not open vocab file");
        exit(EXIT_FAILURE);
    }

    char line[1024];
    while (fgets(line, sizeof(line), file))
    {

        char *hex_token = strtok(line, " == ");
        char *value_str = strtok(NULL, " == ");
        if (!hex_token || !value_str)
        {
            fprintf(stderr, "Invalid line format in vocab file.\n");
            continue;
        }

        size_t hex_len = strlen(hex_token);
        size_t decoded_string_len = hex_len / 4 + 1;
        char *decoded_string = malloc(decoded_string_len);
        if (!decoded_string)
        {
            perror("Memory allocation failed for decoded string");
            exit(EXIT_FAILURE);
        }

        const char *pos = hex_token;
        size_t char_index = 0;
        while (pos[0] == '0' && pos[1] == 'x')
        {
            unsigned int byte_value;
            if (sscanf(pos, "0x%2X", &byte_value) != 1)
            {
                fprintf(stderr, "Failed to parse hex byte: %s\n", pos);
                free(decoded_string);
                continue;
            }
            decoded_string[char_index++] = (char)byte_value;
            pos += 4;
        }
        decoded_string[char_index] = '\0';

        char *key = strdup(decoded_string);
        if (!key)
        {
            perror("Memory allocation failed for key string");
            free(decoded_string);
            exit(EXIT_FAILURE);
        }
        int value = atoi(value_str);

        hashmap_set(vocab_encode, &(struct Token){.key = key, .value = value});

        free(decoded_string);
    }

    fclose(file);

    initialized_encode = true;

    return Py_None;
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

    encode(text, vocab_encode, pattern, tokens, &tokens_size);

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

static PyObject *p_initalize_decode(PyObject *self, PyObject *args)
{
    char *vocab_file_path; // full file path for now

    if (!PyArg_ParseTuple(args, "si", &vocab_file_path, &vocab_size_decode))
        return NULL;

    vocab_decode = malloc(vocab_size_decode * sizeof(char *));

    if (!vocab_decode)
    {
        fprintf(stderr, "Memory allocation failed for vocab decode array\n");
        return NULL;
    }

    FILE *file = fopen(vocab_file_path, "r");
    if (!file)
    {
        fprintf(stderr, "Could not open vocab file!\n");
        return NULL;
    }

    char line[256];
    while (fgets(line, sizeof(line), file))
    {
        char *hex_str = strdup(strtok(line, " == "));
        char *value_str = strtok(NULL, " == ");

        if (hex_str && value_str)
        {
            int value = atoi(value_str);

            vocab_decode[value] = malloc((strlen(hex_str) * sizeof(char)) / 2);

            char ascii_str[buffer_size];

            hex_str_to_ascii(hex_str, ascii_str);

            strcpy(vocab_decode[value], ascii_str);
        }
    }

    fclose(file);

    initialized_decode = true;

    return Py_None;
}

static PyObject *p_decode(PyObject *self, PyObject *args)
{

    if (!initialized_decode)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "Vocabulary is not initialized for decoding. "
            "Call 'initialize_decode' function first."
        );
        return NULL;
    }
    if (!vocab_size_decode)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "Vocab size is not properly set during initialization. "
            "Please try again."
        );
        return NULL;
    }

    PyObject *tokens;

    if (!PyArg_ParseTuple(args, "O", &tokens))
        return NULL;

    if (!PyList_Check(tokens))
    {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of integers");
        return NULL;
    }

    PyObject *result = decode(tokens, vocab_decode, vocab_size_decode);

    return result;
}

static PyMethodDef huTokenMethods[] = {
    {"bpe_train", p_bpe_train, METH_VARARGS, "BPE training"},
    {"initialize_encode", p_initalize_encode, METH_VARARGS, "Initalize tokenizer encoder"},
    {"initialize_decode", p_initalize_decode, METH_VARARGS, "Initalize tokenizer decoder"},
    {"encode", p_encode, METH_VARARGS, "Encodes string"},
    {"decode", p_decode, METH_VARARGS, "Decodes list of ints"},
    {NULL, NULL, 0, NULL} // signal end of functions
};

static struct PyModuleDef huToken = {
    PyModuleDef_HEAD_INIT,
    "huToken",
    "hutoken module description",
    -1,
    huTokenMethods};

PyMODINIT_FUNC PyInit_hutoken(void)
{
    return PyModule_Create(&huToken);
}