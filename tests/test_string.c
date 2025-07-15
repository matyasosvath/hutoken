#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/string.h"

#define RUN_TEST(test)                          \
    do {                                        \
        printf("Running test: %s...\n", #test); \
        test();                                 \
    } while (0)

void test_init_small_string(void) {
    struct String str;

    assert(string_init(&str, "hello") == STRING_SUCCESS);
    assert(str.is_large == false);
    assert(string_len(&str) == 5);
    assert(strcmp(string_c_str(&str), "hello") == 0);

    string_release(&str);
}

void test_init_large_string(void) {
    const char* large_str =
        "This string is intentionally made very long to "
        "exceed SSO buffer.";
    size_t large_str_len = strlen(large_str);
    assert(large_str_len > STRING_SSO_MAX_LEN);

    struct String str;

    assert(string_init(&str, large_str) == STRING_SUCCESS);
    assert(str.is_large == true);
    assert(string_len(&str) == large_str_len);
    assert(strcmp(string_c_str(&str), large_str) == 0);

    string_release(&str);
}

void test_init_empty_string(void) {
    struct String str;

    assert(string_init(&str, "") == STRING_SUCCESS);
    assert(str.is_large == false);
    assert(string_len(&str) == 0);
    assert(strcmp(string_c_str(&str), "") == 0);

    string_release(&str);
}

void test_init_null_string(void) {
    struct String str;

    assert(string_init(&str, NULL) == STRING_SUCCESS);
    assert(str.is_large == false);
    assert(string_len(&str) == 0);
    assert(strcmp(string_c_str(&str), "") == 0);

    string_release(&str);
}

void test_init_sso_max_len_is_small(void) {
    struct String str;
    char sso_max[STRING_SSO_MAX_LEN + 1];
    memset(sso_max, 'a', STRING_SSO_MAX_LEN);
    sso_max[STRING_SSO_MAX_LEN] = '\0';

    assert(string_init(&str, sso_max) == STRING_SUCCESS);
    assert(str.is_large == false);
    assert(string_len(&str) == STRING_SSO_MAX_LEN);

    string_release(&str);
}

void test_init_sso_max_len_plus_one_is_large(void) {
    struct String str;
    char large_str[STRING_SSO_MAX_LEN + 2];
    memset(large_str, 'a', STRING_SSO_MAX_LEN + 1);
    large_str[STRING_SSO_MAX_LEN + 1] = '\0';

    assert(string_init(&str, large_str) == STRING_SUCCESS);
    assert(str.is_large == true);
    assert(string_len(&str) == STRING_SSO_MAX_LEN + 1);

    string_release(&str);
}

void test_with_capacity_small(void) {
    struct String str;

    assert(string_with_capacity(&str, 10) == STRING_SUCCESS);
    assert(str.is_large == false);
    assert(string_len(&str) == 0);

    string_release(&str);
}

void test_with_capacity_sso_exact_is_small(void) {
    struct String str;

    assert(string_with_capacity(&str, STRING_SSO_MAX_LEN) == STRING_SUCCESS);
    assert(str.is_large == false);
    assert(string_len(&str) == 0);

    string_release(&str);
}

void test_with_capacity_large(void) {
    struct String str;

    assert(string_with_capacity(&str, 100) == STRING_SUCCESS);
    assert(str.is_large == true);
    assert(string_len(&str) == 0);
    assert(str.data.large.capacity == 100);

    string_release(&str);
}

void test_append_to_small_remains_small(void) {
    struct String str;
    string_init(&str, "hu");

    assert(string_append(&str, "token") == STRING_SUCCESS);
    assert(str.is_large == false);
    assert(string_len(&str) == 7);
    assert(strcmp(string_c_str(&str), "hutoken") == 0);

    string_release(&str);
}

void test_append_to_small_becomes_large(void) {
    struct String str;
    char almost_full[STRING_SSO_MAX_LEN];
    memset(almost_full, 'a', STRING_SSO_MAX_LEN - 1);
    almost_full[STRING_SSO_MAX_LEN - 1] = '\0';

    string_init(&str, almost_full);
    assert(str.is_large == false);

    assert(string_append(&str, "123") == STRING_SUCCESS);
    assert(str.is_large == true);
    assert(string_len(&str) == STRING_SSO_MAX_LEN - 1 + 3);
    assert(strncmp(string_c_str(&str), almost_full, STRING_SSO_MAX_LEN - 1) ==
           0);
    assert(strcmp(string_c_str(&str) + STRING_SSO_MAX_LEN - 1, "123") == 0);

    string_release(&str);
}

void test_append_to_large_remains_large(void) {
    struct String str;
    string_with_capacity(&str, 100);
    string_append(&str, "hello ");

    assert(string_append(&str, "world!") == STRING_SUCCESS);
    assert(str.is_large == true);
    assert(string_len(&str) == 12);
    assert(strcmp(string_c_str(&str), "hello world!") == 0);

    string_release(&str);
}

void test_append_to_empty(void) {
    struct String str;
    string_init(&str, "");

    assert(string_append(&str, "non-empty") == STRING_SUCCESS);
    assert(string_len(&str) == 9);
    assert(strcmp(string_c_str(&str), "non-empty") == 0);

    string_release(&str);
}

void test_append_empty_string_is_noop(void) {
    struct String str;
    string_init(&str, "hutoken");

    assert(string_append(&str, "") == STRING_SUCCESS);
    assert(string_len(&str) == 7);
    assert(strcmp(string_c_str(&str), "hutoken") == 0);

    string_release(&str);
}

void test_init_handles_null_struct_ptr(void) {
    assert(string_init(NULL, "abc") == STRING_INVALID_ARGUMENT);
}

void test_with_capacity_handles_null_struct_ptr(void) {
    assert(string_with_capacity(NULL, 10) == STRING_INVALID_ARGUMENT);
}

void test_append_handles_null_struct_ptr(void) {
    assert(string_append(NULL, "abc") == STRING_INVALID_ARGUMENT);
}

void test_append_handles_null_to_append_ptr(void) {
    struct String str;
    string_init(&str, "test");

    assert(string_append(&str, NULL) == STRING_INVALID_ARGUMENT);

    string_release(&str);
}

void test_len_handles_null_struct_ptr(void) {
    assert(string_len(NULL) == 0);
}

void test_c_str_handles_null_struct_ptr(void) {
    assert(strcmp(string_c_str(NULL), "") == 0);
}

void test_release_handles_null_struct_ptr(void) {
    string_release(NULL);  // would cause segfault if doesn't work
}

int main(void) {
    puts("Starting string tests.\n");

    RUN_TEST(test_init_small_string);
    RUN_TEST(test_init_large_string);
    RUN_TEST(test_init_empty_string);
    RUN_TEST(test_init_null_string);
    RUN_TEST(test_init_sso_max_len_is_small);
    RUN_TEST(test_init_sso_max_len_plus_one_is_large);
    RUN_TEST(test_with_capacity_small);
    RUN_TEST(test_with_capacity_sso_exact_is_small);
    RUN_TEST(test_with_capacity_large);
    RUN_TEST(test_append_to_small_remains_small);
    RUN_TEST(test_append_to_small_becomes_large);
    RUN_TEST(test_append_to_large_remains_large);
    RUN_TEST(test_append_to_empty);
    RUN_TEST(test_append_empty_string_is_noop);
    RUN_TEST(test_init_handles_null_struct_ptr);
    RUN_TEST(test_with_capacity_handles_null_struct_ptr);
    RUN_TEST(test_append_handles_null_struct_ptr);
    RUN_TEST(test_append_handles_null_to_append_ptr);
    RUN_TEST(test_len_handles_null_struct_ptr);
    RUN_TEST(test_c_str_handles_null_struct_ptr);
    RUN_TEST(test_release_handles_null_struct_ptr);

    puts("\nAll tests passed successfully!");

    return EXIT_SUCCESS;
}
