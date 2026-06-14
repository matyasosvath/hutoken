#!/usr/bin/env bash
set -euo pipefail

FILE_PATHS=("./1_MB_.txt")
THREADS=(1 2 4 8 16 32 64)
OUTPUT_FILE="ablation_results.txt"
ITER=1000
BENCHMARK_SCRIPT="scripts/benchmark.py"
PYTHON="python3"

ABLATION_CONFIGS=(
  "baseline:"
  "disable_arena:--disable-arena"
  "disable_pretokenizer:--disable-pretokenizer"
  "disable_aho_corasick:--disable-aho-corasick"
  "disable_bpe_optimized:--disable-bpe-optimized"
)

> "$OUTPUT_FILE"

echo "Running ablation benchmark suite..." | tee -a "$OUTPUT_FILE"

for FILE_PATH in "${FILE_PATHS[@]}"; do
  for t in "${THREADS[@]}"; do
    echo "================================================================" | tee -a "$OUTPUT_FILE"
    echo "File: $FILE_PATH, Threads: $t" | tee -a "$OUTPUT_FILE"
    echo "================================================================" | tee -a "$OUTPUT_FILE"

    for config in "${ABLATION_CONFIGS[@]}"; do
      name="${config%%:*}"
      flag="${config#*:}"

      echo "---- Run: $name ----" | tee -a "$OUTPUT_FILE"
      if [[ -z "$flag" ]]; then
        "$PYTHON" "$BENCHMARK_SCRIPT" --file-path "$FILE_PATH" --thread-number "$t" --iter "$ITER" | tee -a "$OUTPUT_FILE"
      else
        "$PYTHON" "$BENCHMARK_SCRIPT" --file-path "$FILE_PATH" --thread-number "$t" --iter "$ITER" "$flag" | tee -a "$OUTPUT_FILE"
      fi
      echo >> "$OUTPUT_FILE"
    done

    echo >> "$OUTPUT_FILE"
  done
done

echo "Ablation benchmark complete. Results written to $OUTPUT_FILE." | tee -a "$OUTPUT_FILE"
