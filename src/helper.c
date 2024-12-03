#ifndef HELPER
#define HELPER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define VISUALIZE 1
#define TEXT_SIZE_INCREMENT 50


typedef struct {
    char *start;
    char *end;
} Boundary;


void visualize(int arr[], char *text, int n)
{

    if (VISUALIZE)
    {
        printf("Processing:\n");
        for (int i = 0; i < n; i++)
        {
            printf("%c ", text[i]);
        }
        printf("\n");
        for (int i = 0; i < n; i++)
        {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
}

void hex_str_to_ascii(const char *hex_str, char *ascii_str) {
    int i = 0;
    while (*hex_str != '\0') {
        if (*hex_str == '0' && *(hex_str + 1) == 'x') {
            hex_str += 2;  // skip "0x"

            char hexValue[3] = { hex_str[0], hex_str[1], '\0' };
            int charValue = (int)strtol(hexValue, NULL, 16);

            ascii_str[i++] = (char)charValue;
            hex_str += 2;
        }
    }
    ascii_str[i] = '\0';
}

#endif