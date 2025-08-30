#!/bin/bash

FILE_PATHS=("benchmark.txt" "./1_MB.txt" "./10_MB.txt" "./100_MB.txt" "./1000_MB.txt")
THREADS=(1 2 4 8 16 32 64)
OUTPUT_FILE="benchmark_results.txt"

# Clear the file before writing
> "$OUTPUT_FILE"

for FILE_PATH in "${FILE_PATHS[@]}"
do
    for t in "${THREADS[@]}"
    do
        echo "==== Running with --file-path $FILE_PATH --thread-number $t ====" | tee -a "$OUTPUT_FILE"
        python3 scripts/benchmark.py --file-path "$FILE_PATH" --thread-number $t --iter 1000 | tee -a "$OUTPUT_FILE"
        echo -e "\n" >> "$OUTPUT_FILE"
    done
done
