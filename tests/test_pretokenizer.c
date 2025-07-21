#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/pretokenizer.h"

#define RUN_TEST(test)                          \
    do {                                        \
        printf("Running test: %s...\n", #test); \
        test();                                 \
    } while (0)

void test_null_input_string(void) {
    const char* replacements[256] = {NULL};
    char* result = pretokenizer_encode(NULL, replacements, NULL);

    assert(result == NULL);
}

void test_no_replacements_needed(void) {
    const char* text = "hello world";
    const char* replacements[256] = {NULL};

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "hello world") == 0);

    free(result);
}

void test_single_replacement_at_start(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['a'] = "Alpha";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "Alphapple") == 0);

    free(result);
}

void test_single_replacement_at_end(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['e'] = "End";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "applEnd") == 0);

    free(result);
}

void test_single_replacement_in_middle(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['p'] = "P";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "aPPle") == 0);

    free(result);
}

void test_multiple_different_replacements(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['a'] = "ab";
    replacements['e'] = "ef";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "abpplef") == 0);

    free(result);
}

void test_multiple_occurrences_of_same_char(void) {
    const char* text = "banana";
    const char* replacements[256] = {NULL};
    replacements['a'] = "o";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "bonono") == 0);

    free(result);
}

void test_replacement_with_empty_string(void) {
    const char* text = "hello";
    const char* replacements[256] = {NULL};
    replacements['l'] = "";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "heo") == 0);

    free(result);
}

void test_empty_input_string(void) {
    const char* text = "";
    const char* replacements[256] = {NULL};
    replacements['a'] = "b";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "") == 0);

    free(result);
}

void test_all_chars_are_replaced(void) {
    const char* text = "abc";
    const char* replacements[256] = {NULL};
    replacements['a'] = "1";
    replacements['b'] = "22";
    replacements['c'] = "333";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "122333") == 0);

    free(result);
}

void test_replacement_with_single_char(void) {
    const char* text = "test";
    const char* replacements[256] = {NULL};
    replacements['t'] = "T";

    char* result = pretokenizer_encode(text, replacements, NULL);

    assert(strcmp(result, "TesT") == 0);

    free(result);
}

void test_with_prefix_and_replacements(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['a'] = "A";
    replacements['e'] = "E";

    char* result = pretokenizer_encode(text, replacements, "Juicy ");

    assert(strcmp(result, "Juicy ApplE") == 0);

    free(result);
}

void test_with_prefix_and_empty_string_replacement(void) {
    const char* text = "-abc-";
    const char* replacements[256] = {NULL};
    replacements['-'] = "";

    char* result = pretokenizer_encode(text, replacements, "Prefix:");

    assert(strcmp(result, "Prefix:abc") == 0);

    free(result);
}

void test_with_prefix_and_empty_input_string(void) {
    const char* text = "";
    const char* replacements[256] = {NULL};

    char* result = pretokenizer_encode(text, replacements, "Start:");

    assert(strcmp(result, "Start:") == 0);

    free(result);
}

void test_with_empty_string_as_prefix(void) {
    const char* text = "test";
    const char* replacements[256] = {NULL};

    char* result = pretokenizer_encode(text, replacements, "");

    assert(strcmp(result, "test") == 0);

    free(result);
}

int main(void) {
    puts("Starting string tests.\n");

    RUN_TEST(test_null_input_string);
    RUN_TEST(test_no_replacements_needed);
    RUN_TEST(test_single_replacement_at_start);
    RUN_TEST(test_single_replacement_at_end);
    RUN_TEST(test_single_replacement_in_middle);
    RUN_TEST(test_multiple_different_replacements);
    RUN_TEST(test_multiple_occurrences_of_same_char);
    RUN_TEST(test_replacement_with_empty_string);
    RUN_TEST(test_empty_input_string);
    RUN_TEST(test_all_chars_are_replaced);
    RUN_TEST(test_replacement_with_single_char);
    RUN_TEST(test_with_prefix_and_replacements);
    RUN_TEST(test_with_prefix_and_empty_string_replacement);
    RUN_TEST(test_with_prefix_and_empty_input_string);
    RUN_TEST(test_with_empty_string_as_prefix);

    puts("\nAll tests passed successfully!");

    return EXIT_SUCCESS;
}
