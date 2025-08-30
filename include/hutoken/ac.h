#ifndef HUTOKEN_AC_H
#define HUTOKEN_AC_H

#include <stdbool.h>
#include <stddef.h>

#define AC_ALPHABET_SIZE 256

struct ACNode {
    struct ACNode* children[AC_ALPHABET_SIZE];
    struct ACNode* failure_link;
    int output_value;
    size_t pattern_len;
};

struct ACAutomaton {
    struct ACNode* root;
};

struct ACAutomaton* ac_automaton_create(void);
bool ac_automaton_add_string(struct ACAutomaton* automaton,
                             const char* pattern,
                             int output_value);
void ac_automaton_build_failure_links(struct ACAutomaton* automaton);
void ac_automaton_free(struct ACAutomaton* automaton);

#endif
