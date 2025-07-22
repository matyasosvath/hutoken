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
    char* result_encoded = pretokenizer_encode(NULL, replacements, NULL);
    char* result_decoded = pretokenizer_decode(NULL, replacements, NULL);

    assert(result_encoded == NULL);
    assert(result_decoded == NULL);
}

void test_no_replacements_needed(void) {
    const char* text = "hello world";
    const char* replacements[256] = {NULL};

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "hello world") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_single_replacement_at_start(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['a'] = "Alpha";

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "Alphapple") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_single_replacement_at_end(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['e'] = "End";

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "applEnd") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_single_replacement_in_middle(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['p'] = "P";

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "aPPle") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_multiple_different_replacements(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['a'] = "ab";
    replacements['e'] = "ef";

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "abpplef") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_multiple_occurrences_of_same_char(void) {
    const char* text = "banana";
    const char* replacements[256] = {NULL};
    replacements['a'] = "o";

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "bonono") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

// won't happen in practice
// decoding will not be able to handle this
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

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_all_chars_are_replaced(void) {
    const char* text = "abc";
    const char* replacements[256] = {NULL};
    replacements['a'] = "1";
    replacements['b'] = "22";
    replacements['c'] = "333";

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "122333") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_replacement_with_single_char(void) {
    const char* text = "test";
    const char* replacements[256] = {NULL};
    replacements['t'] = "T";

    char* result_encoded = pretokenizer_encode(text, replacements, NULL);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, NULL);

    assert(strcmp(result_encoded, "TesT") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_with_prefix_and_replacements(void) {
    const char* text = "apple";
    const char* replacements[256] = {NULL};
    replacements['a'] = "A";
    replacements['e'] = "E";
    const char* prefix = "Juicy ";

    char* result_encoded = pretokenizer_encode(text, replacements, prefix);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, prefix);

    assert(strcmp(result_encoded, "Juicy ApplE") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

// won't happen in practice
// decoding will not be able to handle this
void test_with_prefix_and_empty_string_replacement(void) {
    const char* text = "-abc-";
    const char* replacements[256] = {NULL};
    replacements['-'] = "";
    const char* prefix = "Prefix:";

    char* result = pretokenizer_encode(text, replacements, prefix);

    assert(strcmp(result, "Prefix:abc") == 0);

    free(result);
}

void test_with_prefix_and_empty_input_string(void) {
    const char* text = "";
    const char* replacements[256] = {NULL};
    const char* prefix = "Start:";

    char* result_encoded = pretokenizer_encode(text, replacements, prefix);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, prefix);

    assert(strcmp(result_encoded, "Start:") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
}

void test_with_empty_string_as_prefix(void) {
    const char* text = "test";
    const char* replacements[256] = {NULL};
    const char* prefix = "";

    char* result_encoded = pretokenizer_encode(text, replacements, prefix);
    char* result_decoded = pretokenizer_decode(result_encoded, replacements, prefix);

    assert(strcmp(result_encoded, "test") == 0);
    assert(strcmp(result_decoded, text) == 0);

    free(result_encoded);
    free(result_decoded);
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
