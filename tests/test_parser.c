#include <assert.h>
#include <locale.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hutoken/parser.h"

const char* BPE_REGEX_PATTERN =
    "[ ]?[A-Za-záéíóúőűüöÁÉÍÓÚŐÜŰÖ]+|[ ]?[0-9]+|[ "
    "]?[^[:space:][:alpha:][:digit:]]+|[ ]+";

void run_comparison_test(const char* text, regex_t* compiled_regex) {
    printf("Comparing on text: \"%.40s%s\"\n", text,
           strlen(text) > 40 ? "..." : "");

    struct ParserState custom_parser_state = parser_init(text);
    const char* regex_cursor = text;

    int token_count = 0;
    while (1) {
        token_count++;

        struct TokenSlice custom_token;
        bool custom_parser_has_token =
            parser_next_token(&custom_parser_state, &custom_token);

        regmatch_t regex_match[1];
        size_t oracle_token_len = 0;
        bool regex_found_match =
            (regexec(compiled_regex, regex_cursor, 1, regex_match, 0) == 0);

        if (regex_found_match && regex_match[0].rm_so == 0) {
            oracle_token_len = regex_match[0].rm_eo - regex_match[0].rm_so;
        } else if (*regex_cursor != '\0') {
            oracle_token_len = 1;
        }

        bool oracle_has_token = (oracle_token_len > 0);

        assert(custom_parser_has_token == oracle_has_token);

        if (!custom_parser_has_token) {
            break;
        }

        assert(custom_token.length == oracle_token_len);

        assert(strncmp(custom_token.start, regex_cursor, custom_token.length) ==
               0);

        regex_cursor += oracle_token_len;

        assert(token_count < 1000);
    }
    printf("... OK\n");
}

int main(void) {
    if (setlocale(LC_ALL, "en_US.UTF-8") == NULL) {
        (void)fprintf(stderr,
                      "Warning: Could not set locale to en_US.UTF-8. Unicode "
                      "tests might fail.\n");
    }

    regex_t bpe_regex;
    int reg_comp_status = regcomp(&bpe_regex, BPE_REGEX_PATTERN, REG_EXTENDED);
    if (reg_comp_status != 0) {
        char error_buffer[100];
        regerror(reg_comp_status, &bpe_regex, error_buffer,
                 sizeof(error_buffer));
        fprintf(stderr, "Regex compilation failed: %s\n", error_buffer);
        return EXIT_FAILURE;
    }

    const char* test_cases[] = {"",
                                "Hello",
                                " Hello",
                                "árvíztűrő",
                                " árvíztűrő",
                                "ÁrvíztűrőTükör",
                                "12345",
                                " 123",
                                "!!!@#$",
                                " !!!",
                                "!@#$%^&*()_+",
                                "€",
                                "😂",
                                " ",
                                "   ",

                                "word123",
                                "word.!_",
                                "123word",
                                "123.!",
                                ".!word",
                                ".!123",

                                "\t\n\r\f\v",
                                "A  B   C",
                                "word ",
                                " First Second",

                                "Hello world 123. End!",
                                "árvíztűrő tükörfúrógép.",
                                "This is a test 123. With some special chars: "
                                "!@# and spaces. árvíztűrő tükörfúrógép!"};

    puts("Starting BPE Parser Golden Master Tests.\n");

    size_t num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (size_t i = 0; i < num_test_cases; ++i) {
        run_comparison_test(test_cases[i], &bpe_regex);
    }

    regfree(&bpe_regex);

    puts("\nAll parser tests passed successfully!");
    return EXIT_SUCCESS;
}
