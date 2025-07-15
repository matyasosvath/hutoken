#include "hutoken/string.h"

#include <stdlib.h>
#include <string.h>

static enum StringError grow(struct String* str, size_t needed_len);

enum StringError string_init(struct String* str, const char* init) {
    if (!str) {
        return STRING_INVALID_ARGUMENT;
    }

    size_t len = init ? strlen(init) : 0;

    if (len <= STRING_SSO_MAX_LEN) {
        str->is_large = false;
        if (init) {
            memcpy(str->data.small, init, len);
        }
        str->data.small[STRING_SSO_MAX_LEN] = STRING_SSO_MAX_LEN - len;
        str->data.small[len] = '\0';
    } else {
        str->is_large = true;
        str->data.large.buf = (char*)malloc(len + 1);
        if (!str->data.large.buf) {
            return STRING_ALLOC_ERROR;
        }

        memcpy(str->data.large.buf, init, len);
        str->data.large.buf[len] = '\0';
        str->data.large.len = len;
        str->data.large.capacity = len;
    }

    return STRING_SUCCESS;
}

enum StringError string_with_capacity(struct String* str, size_t capacity) {
    if (!str) {
        return STRING_INVALID_ARGUMENT;
    }

    if (capacity <= STRING_SSO_MAX_LEN) {
        str->is_large = false;
        str->data.small[STRING_SSO_MAX_LEN] = STRING_SSO_MAX_LEN;
        str->data.small[0] = '\0';
    } else {
        str->is_large = true;
        str->data.large.buf = (char*)malloc(capacity + 1);
        if (!str->data.large.buf) {
            return STRING_ALLOC_ERROR;
        }
        str->data.large.buf[0] = '\0';
        str->data.large.len = 0;
        str->data.large.capacity = capacity;
    }

    return STRING_SUCCESS;
}

void string_release(struct String* str) {
    if (str && str->is_large) {
        free(str->data.large.buf);
    }
}

const char* string_c_str(const struct String* str) {
    if (!str) {
        return "";
    }

    return str->is_large ? str->data.large.buf : str->data.small;
}

size_t string_len(const struct String* str) {
    if (!str) {
        return 0;
    }

    if (str->is_large) {
        return str->data.large.len;
    }

    return STRING_SSO_MAX_LEN - str->data.small[STRING_SSO_MAX_LEN];
}

enum StringError string_append(struct String* str, const char* to_append) {
    if (!str || !to_append) {
        return STRING_INVALID_ARGUMENT;
    }

    size_t current_len = string_len(str);
    size_t append_len = strlen(to_append);
    if (append_len == 0) {
        return STRING_SUCCESS;
    }

    size_t needed_len = current_len + append_len;
    if (grow(str, needed_len) != STRING_SUCCESS) {
        return STRING_ALLOC_ERROR;
    }

    char* buffer = str->is_large ? str->data.large.buf : str->data.small;
    memcpy(buffer + current_len, to_append, append_len);
    buffer[needed_len] = '\0';

    if (str->is_large) {
        str->data.large.len = needed_len;
    } else {
        str->data.small[STRING_SSO_MAX_LEN] = STRING_SSO_MAX_LEN - needed_len;
    }

    return STRING_SUCCESS;
}

enum StringError string_clear(struct String* str) {
    if (!str) {
        return STRING_INVALID_ARGUMENT;
    }

    if (str->is_large) {
        str->data.large.len = 0;
        str->data.large.buf[0] = '\0';
    } else {
        str->data.small[0] = '\0';
        str->data.small[STRING_SSO_MAX_LEN] = STRING_SSO_MAX_LEN;
    }

    return STRING_SUCCESS;
}

static enum StringError grow(struct String* str, size_t needed_len) {
    size_t capacity =
        str->is_large ? str->data.large.capacity : STRING_SSO_MAX_LEN;
    if (needed_len <= capacity) {
        return STRING_SUCCESS;
    }

    size_t new_capacity = capacity < 8 ? 8 : capacity;
    while (new_capacity <= needed_len) {
        new_capacity *= 2;
    }

    if (str->is_large) {
        char* new_buf = realloc(str->data.large.buf, new_capacity + 1);
        if (!new_buf) {
            return STRING_ALLOC_ERROR;
        }

        str->data.large.buf = new_buf;
        str->data.large.capacity = new_capacity;
    } else {
        char* new_buf = malloc(new_capacity + 1);
        if (!new_buf) {
            return STRING_ALLOC_ERROR;
        }

        size_t old_len = string_len(str);
        memcpy(new_buf, str->data.small, old_len);

        str->is_large = true;
        str->data.large.buf = new_buf;
        str->data.large.len = old_len;
        str->data.large.capacity = new_capacity;
    }

    return STRING_SUCCESS;
}
