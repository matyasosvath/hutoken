#ifndef HUTOKEN_STRING_H
#define HUTOKEN_STRING_H

#include <stdbool.h>
#include <stddef.h>

#define STRING_SSO_MAX_LEN (sizeof(((struct String*)0)->data.small) - 1)

struct String {
    bool is_large;
    union {
        struct LargeString {
            size_t len;
            size_t capacity;
            char* buf;
        } large;
        char small[sizeof(struct LargeString) - 1];
    } data;
};

enum StringError {
    STRING_SUCCESS,
    STRING_ALLOC_ERROR,
    STRING_INVALID_ARGUMENT,
};

enum StringError string_init(struct String* str, const char* init);
enum StringError string_with_capacity(struct String* str, size_t capacity);
void string_release(struct String* str);

const char* string_c_str(const struct String* str);
size_t string_len(const struct String* str);
enum StringError string_append(struct String* str, const char* to_append);

#endif
