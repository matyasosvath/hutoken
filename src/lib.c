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
int vocab_size_decode;



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

static PyObject *p_initialize_decode(PyObject *self, PyObject *args) {

    char *vocab_file_path;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "si", &vocab_file_path, &vocab_size_decode)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments. Expected a string (vocab_file_path) and an integer (vocab_size_decode).");
        return NULL;
    }

    // Check if vocab_size_decode is valid
    if (vocab_size_decode <= 0) {
        PyErr_SetString(PyExc_ValueError, "vocab_size_decode must be greater than zero.");
        return NULL;
    }

    // If Python garbage collector collects the string, vocab_file_path could become invalid.
    char *vocab_file_path_copy = strdup(vocab_file_path);
    if (!vocab_file_path_copy) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for vocab file path.");
        return NULL;
    }

    // Allocate memory for vocab_decode array
    vocab_decode = malloc(vocab_size_decode * sizeof(char *));
    if (!vocab_decode) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for vocab_decode array.");
        free(vocab_file_path_copy);
        return NULL;
    }

    // Initialize all entries to NULL
    for (int i = 0; i < vocab_size_decode; i++)
        vocab_decode[i] = NULL;

    // Open vocab file
    FILE *file = fopen(vocab_file_path_copy, "r");
    if (!file) {
        PyErr_SetString(PyExc_FileNotFoundError, "Could not open vocab file.");
        free(vocab_decode);
        free(vocab_file_path_copy);
        return NULL;
    }

    // Read vocab file line by line
    char line[256];
    while (fgets(line, sizeof(line), file))
    {
        char hex_str[256];
        int value;

        // Parse line and check if format is correct
        if (sscanf(line, "%255s == %d", hex_str, &value) != 2) {
            PyErr_SetString(PyExc_ValueError, "Invalid format in vocab file.");
            fclose(file);
            free(vocab_file_path_copy);
            for (int i = 0; i < vocab_size_decode; i++) {
                free(vocab_decode[i]);
            }
            free(vocab_decode);
            return NULL;
        }
        
        // Check if value is within bounds
        if (value < 0 || value >= vocab_size_decode) {
            PyErr_SetString(PyExc_ValueError, "Invalid vocab index in vocab file.");
            fclose(file);
            free(vocab_file_path_copy);
            for (int i = 0; i < vocab_size_decode; i++) {
                free(vocab_decode[i]);
            }
            free(vocab_decode);
            return NULL;
        }

        // Allocate memory for ASCII string
        size_t ascii_str_len = strlen(hex_str) / 4 + 1;     // Each hex byte is 4 characters (e.g., "0x41"), plus 1 for the null t.
        char *ascii_str = malloc(ascii_str_len);            // Allocate memory for ASCII string
        if (!ascii_str) {                                   // Check if memory allocation failed
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for ascii_str.");
            fclose(file);
            free(vocab_file_path_copy);
            for (int i = 0; i < vocab_size_decode; i++)
                free(vocab_decode[i]);
            free(vocab_decode);
            free(ascii_str);
            return NULL;
        }

        // Convert hex string to ASCII
        if (!hex_str_to_ascii(hex_str, ascii_str) || ascii_str[0] == '\0') {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert hex string to ASCII.");
            fclose(file);
            free(vocab_file_path_copy);
            for (int i = 0; i < vocab_size_decode; i++) {
                free(vocab_decode[i]);
            }
            free(vocab_decode);
            free(ascii_str);
            return NULL;
        }

        // Copy ASCII string to vocab_decode
        vocab_decode[value] = strdup(ascii_str);
        free(ascii_str);
        if (!vocab_decode[value]) {
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for vocab entry.");
            fclose(file);
            free(vocab_file_path_copy);
            for (int i = 0; i < vocab_size_decode; i++)
                free(vocab_decode[i]);
            free(vocab_decode);
            free(ascii_str);
            return NULL;
        }
    }

    free(vocab_file_path_copy);
    fclose(file);

    initialized_decode = true;

    Py_RETURN_NONE;     // Return None object to Python code to indicate success
}


static PyObject *p_decode(PyObject *self, PyObject *args) {

    // Check if vocabulary is initialized for decoding
    if (!initialized_decode) {
        PyErr_SetString(PyExc_RuntimeError,
            "Vocabulary is not initialized for decoding. "
            "Call 'initialize_decode' function first."
        );
        return NULL;
    }

    // Check if vocab_size_decode is properly set
    if (!vocab_size_decode) {
        PyErr_SetString(PyExc_RuntimeError,
            "Vocab size is not properly set during initialization. "
            "Please try again."
        );
        return NULL;
    }

    // Initialize variables
    PyObject *tokens;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "O", &tokens)) {
        PyErr_SetString(PyExc_TypeError, "Failed to parse arguments. Expected a single list of tokens.");
        return NULL;
    }

    // Check if tokens is a list
    if (!PyList_Check(tokens)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of integers");
        return NULL;
    }

    // Check if token list is empty
    Py_ssize_t num_tokens = PyList_Size(tokens);
    if (num_tokens <= 0) {
        PyErr_SetString(PyExc_ValueError, "Token list must not be empty.");
        return NULL;
    }

    // Allocate buffer for decoding result dynamically
    size_t result_size = 1; // Start with 1 for the null terminator
    char *result = malloc(result_size);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed for decoding result.");
        return NULL;
    }
    result[0] = '\0'; // Initialize as an empty string

    // Decode tokens
    for (Py_ssize_t i = 0; i < num_tokens; i++) {
        
        // Check if token is an integer
        PyObject *token_obj = PyList_GetItem(tokens, i);
        if (!PyLong_Check(token_obj)) {
            PyErr_SetString(PyExc_TypeError, "All tokens must be integers.");
            free(result);
            return NULL;
        }

        // Get token value
        int token = (int)PyLong_AsLong(token_obj);                                  // Convert Python object to C integer
        if (token < 0 || token >= vocab_size_decode || !vocab_decode[token]) {      // Check if token is within bounds
            PyErr_SetString(PyExc_ValueError, "Invalid token value or uninitialized vocabulary entry.");
            free(result);
            return NULL;
        }

        // Calculate required size for the new token
        size_t token_len = strlen(vocab_decode[token]);
        size_t new_size = result_size + token_len;

        // Reallocate memory for the result buffer
        char *temp = realloc(result, new_size);
        if (!temp) {
            PyErr_SetString(PyExc_MemoryError, "Memory reallocation failed for decoding result.");
            free(result);
            return NULL;
        }
        // Assign temp to result only after checking for failure
        result = temp;

        // Append token to result
        strncat(result, vocab_decode[token], token_len);
        result_size = new_size;
    }

    // Create Python string from decoding result
    PyObject *py_result = PyUnicode_FromString(result);
    if (!py_result) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Python string from decoding result.");
        free(result);
        return NULL;
    }

    free(result);
    
    return py_result;
}

static PyMethodDef huTokenMethods[] = {
    {"bpe_train", p_bpe_train, METH_VARARGS, "BPE training"},
    {"initialize_encode", p_initalize_encode, METH_VARARGS, "Initalize tokenizer encoder"},
    {"initialize_decode", p_initialize_decode, METH_VARARGS, "Initalize tokenizer decoder"},
    {"encode", p_encode, METH_VARARGS, "Encodes string"},
    {"decode", p_decode, METH_VARARGS, "Decodes list of ints"},
    {NULL, NULL, 0, NULL} // signal end of functions
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