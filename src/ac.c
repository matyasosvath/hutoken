#include "hutoken/ac.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/helper.h"

static struct ACNode* create_node(void);
static void free_node(struct ACNode* node);

struct ACAutomaton* ac_automaton_create(void) {
    struct ACAutomaton* automaton =
        (struct ACAutomaton*)malloc(sizeof(struct ACAutomaton));
    if (!automaton) {
        log_debug("Error: Failed to allocate memory for ACAutomaton.");
        return NULL;
    }
    automaton->root = create_node();
    if (!automaton->root) {
        free(automaton);
        return NULL;
    }
    return automaton;
}

bool ac_automaton_add_string(struct ACAutomaton* automaton,
                             const char* pattern,
                             int output_value) {
    if (!automaton || !pattern) {
        return false;
    }
    struct ACNode* current = automaton->root;
    size_t len = strlen(pattern);

    for (size_t i = 0; i < len; ++i) {
        unsigned char index = (unsigned char)pattern[i];
        if (!current->children[index]) {
            current->children[index] = create_node();
            if (!current->children[index]) {
                return false;
            }
        }
        current = current->children[index];
    }
    current->output_value = output_value;
    current->pattern_len = len;
    return true;
}

void ac_automaton_build_failure_links(struct ACAutomaton* automaton) {
    if (!automaton || !automaton->root) {
        return;
    }

    struct ACNode* root = automaton->root;
    root->failure_link = root;

    struct ACNode** queue =
        (struct ACNode**)malloc(sizeof(struct ACNode*) * AC_ALPHABET_SIZE * 4);
    if (!queue) {
        log_debug(
            "Error: Failed to allocate memory for AC failure link build "
            "queue.");
        return;
    }
    int head = 0;
    int tail = 0;

    for (int i = 0; i < AC_ALPHABET_SIZE; ++i) {
        if (root->children[i]) {
            root->children[i]->failure_link = root;
            queue[tail++] = root->children[i];
        }
    }

    while (head < tail) {
        struct ACNode* current = queue[head++];

        for (int i = 0; i < AC_ALPHABET_SIZE; ++i) {
            struct ACNode* child = current->children[i];
            if (child) {
                struct ACNode* failure_node = current->failure_link;
                while (failure_node != root && !failure_node->children[i]) {
                    failure_node = failure_node->failure_link;
                }

                if (failure_node->children[i]) {
                    child->failure_link = failure_node->children[i];
                } else {
                    child->failure_link = root;
                }

                queue[tail++] = child;
            }
        }
    }
    free((void*)queue);
}

void ac_automaton_free(struct ACAutomaton* automaton) {
    if (!automaton) {
        return;
    }
    free_node(automaton->root);
    free(automaton);
}

static struct ACNode* create_node(void) {
    struct ACNode* node = (struct ACNode*)malloc(sizeof(struct ACNode));
    if (!node) {
        log_debug("Error: Failed to allocate memory for ACNode.");
        return NULL;
    }
    memset((void*)node->children, 0, sizeof(node->children));
    node->failure_link = NULL;
    node->output_value = -1;
    node->pattern_len = 0;
    return node;
}

static void free_node(struct ACNode* node) {
    if (!node) {
        return;
    }
    for (int i = 0; i < AC_ALPHABET_SIZE; ++i) {
        free_node(node->children[i]);
    }
    free(node);
}
