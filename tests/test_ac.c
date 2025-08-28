#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/ac.h"

#define RUN_TEST(test)                          \
    do {                                        \
        printf("Running test: %s...\n", #test); \
        test();                                 \
    } while (0)

void test_create_and_free_automaton(void) {
    struct ACAutomaton* automaton = ac_automaton_create();
    assert(automaton != NULL);
    assert(automaton->root != NULL);
    ac_automaton_free(automaton);
}

void test_add_single_pattern(void) {
    struct ACAutomaton* automaton = ac_automaton_create();
    const char* pattern = "test";
    int value = 42;

    assert(ac_automaton_add_string(automaton, pattern, value) == true);

    struct ACNode* current = automaton->root;
    for (size_t i = 0; i < strlen(pattern); ++i) {
        unsigned char index = (unsigned char)pattern[i];
        assert(current->children[index] != NULL);
        current = current->children[index];
    }
    assert(current->output_value == value);
    assert(current->pattern_len == strlen(pattern));

    ac_automaton_free(automaton);
}

void test_add_multiple_patterns(void) {
    struct ACAutomaton* automaton = ac_automaton_create();
    assert(ac_automaton_add_string(automaton, "apple", 1) == true);
    assert(ac_automaton_add_string(automaton, "banana", 2) == true);

    struct ACNode* current = automaton->root;
    const char* p1 = "apple";
    for (size_t i = 0; i < strlen(p1); ++i) {
        current = current->children[(unsigned char)p1[i]];
    }
    assert(current->output_value == 1);
    assert(current->pattern_len == 5);

    current = automaton->root;
    const char* p2 = "banana";
    for (size_t i = 0; i < strlen(p2); ++i) {
        current = current->children[(unsigned char)p2[i]];
    }
    assert(current->output_value == 2);
    assert(current->pattern_len == 6);

    ac_automaton_free(automaton);
}

void test_add_overlapping_patterns(void) {
    struct ACAutomaton* automaton = ac_automaton_create();
    assert(ac_automaton_add_string(automaton, "he", 1) == true);
    assert(ac_automaton_add_string(automaton, "her", 2) == true);
    assert(ac_automaton_add_string(automaton, "hers", 3) == true);

    struct ACNode* node_h = automaton->root->children['h'];
    assert(node_h != NULL);

    struct ACNode* node_he = node_h->children['e'];
    assert(node_he != NULL);
    assert(node_he->output_value == 1);
    assert(node_he->pattern_len == 2);

    struct ACNode* node_her = node_he->children['r'];
    assert(node_her != NULL);
    assert(node_her->output_value == 2);
    assert(node_her->pattern_len == 3);

    struct ACNode* node_hers = node_her->children['s'];
    assert(node_hers != NULL);
    assert(node_hers->output_value == 3);
    assert(node_hers->pattern_len == 4);

    ac_automaton_free(automaton);
}

void test_build_failure_links_simple(void) {
    struct ACAutomaton* automaton = ac_automaton_create();
    ac_automaton_add_string(automaton, "ab", 1);
    ac_automaton_add_string(automaton, "bc", 2);
    ac_automaton_add_string(automaton, "a", 3);

    ac_automaton_build_failure_links(automaton);

    struct ACNode* root = automaton->root;
    struct ACNode* node_a = root->children['a'];
    struct ACNode* node_b = root->children['b'];
    struct ACNode* node_ab = node_a->children['b'];
    struct ACNode* node_bc = node_b->children['c'];

    assert(node_a->failure_link == root);
    assert(node_b->failure_link == root);
    assert(node_ab->failure_link == node_b);
    assert(node_bc->failure_link == root);

    ac_automaton_free(automaton);
}

void test_build_failure_links_complex(void) {
    struct ACAutomaton* automaton = ac_automaton_create();
    ac_automaton_add_string(automaton, "he", 1);
    ac_automaton_add_string(automaton, "she", 2);
    ac_automaton_add_string(automaton, "his", 3);
    ac_automaton_add_string(automaton, "hers", 4);

    ac_automaton_build_failure_links(automaton);

    struct ACNode* root = automaton->root;
    struct ACNode* node_h = root->children['h'];
    struct ACNode* node_s = root->children['s'];
    struct ACNode* node_sh = node_s->children['h'];
    struct ACNode* node_she = node_sh->children['e'];
    struct ACNode* node_he = node_h->children['e'];

    assert(node_s->failure_link == root);
    assert(node_h->failure_link == root);
    assert(node_sh->failure_link == node_h);
    assert(node_she->failure_link == node_he);

    ac_automaton_free(automaton);
}

void test_add_invalid_patterns(void) {
    struct ACAutomaton* automaton = ac_automaton_create();
    assert(ac_automaton_add_string(automaton, NULL, 1) == false);

    assert(ac_automaton_add_string(automaton, "", 100) == true);
    assert(automaton->root->output_value == 100);
    assert(automaton->root->pattern_len == 0);

    ac_automaton_free(automaton);
}

void test_free_null_automaton(void) {
    ac_automaton_free(NULL);
}

void test_build_on_empty_automaton(void) {
    struct ACAutomaton* automaton = ac_automaton_create();
    ac_automaton_build_failure_links(automaton);
    assert(automaton->root->failure_link == automaton->root);
    ac_automaton_free(automaton);
}

int main(void) {
    puts("Starting Aho-Corasick automaton tests.\n");

    RUN_TEST(test_create_and_free_automaton);
    RUN_TEST(test_add_single_pattern);
    RUN_TEST(test_add_multiple_patterns);
    RUN_TEST(test_add_overlapping_patterns);
    RUN_TEST(test_build_failure_links_simple);
    RUN_TEST(test_build_failure_links_complex);
    RUN_TEST(test_add_invalid_patterns);
    RUN_TEST(test_free_null_automaton);
    RUN_TEST(test_build_on_empty_automaton);

    puts("\nAll Aho-Corasick tests passed successfully!");

    return EXIT_SUCCESS;
}
