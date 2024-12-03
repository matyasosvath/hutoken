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

#endif