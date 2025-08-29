#!/bin/bash

FILE_PATHS=("cel.txt")
THREADS=(1)
OUTPUT_FILE="benchmark_results.txt"

# Clear the file before writing
> "$OUTPUT_FILE"

for FILE_PATH in "${FILE_PATHS[@]}"
do
    for t in "${THREADS[@]}"
    do
        echo "==== Running with --file-path $FILE_PATH --thread-number $t ====" | tee -a "$OUTPUT_FILE"
        python3 scripts/benchmark.py --file-path "$FILE_PATH" --thread-number $t --iter 1 | tee -a "$OUTPUT_FILE"
        echo -e "\n" >> "$OUTPUT_FILE"
    done
done
