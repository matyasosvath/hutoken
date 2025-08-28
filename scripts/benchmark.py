import math
from typing import Any, cast

import time
import pathlib
import argparse
from statistics import mean

import hutoken
hutoken.initialize("openai-community/gpt2")

import tiktoken
enc = tiktoken.get_encoding("gpt2")

import transformers
hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")


__doc__ = """Measure tokenizers speed and performance for a given document."""

def flatten(lst: list[Any]) -> list[Any]:
    flat = []
    for x in lst:
        if isinstance(x, list):
            flat.extend(x)
        else:
            flat.append(x)
    return flat

def split_document(document: str, num_parts: int) -> list[str]:
    text_len = len(document)
    chunk_size = (text_len + num_parts - 1) // num_parts
    chunks = []
    start = 0

    for i in range(num_parts):
        end = min(start + chunk_size, text_len)

        if end < text_len and i < num_parts - 1:
            while end < text_len and document[end] not in (' ', '\n', '\t'):
                end += 1
        next_start = end

        if start < end:
            chunks.append(document[start:end])
            
        start = next_start

    return chunks

def benchmark(document, num_bytes, thread_number):
    document_batches = split_document(document, thread_number)

    
    hutoken.encode("bemelegítés")

    start = time.perf_counter_ns()
    ht_result = hutoken.encode(document) if thread_number == 1 else hutoken.batch_encode(document_batches, num_threads=thread_number)
    end = time.perf_counter_ns()

    ht_perf_result = num_bytes / (end - start) * 1e9


    enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    tt_result = enc.encode(document) if thread_number == 1 else enc.encode_ordinary_batch(document_batches, num_threads=thread_number)
    end = time.perf_counter_ns()

    tt_perf_result = num_bytes / (end - start) * 1e9

    
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    hf_result = hf_enc(document)["input_ids"] if thread_number == 1 else hf_enc(document_batches)["input_ids"]
    end = time.perf_counter_ns()

    hf_perf_result = num_bytes / (end - start) * 1e9

    if not ht_result == tt_result == hf_result:

        print("\n=== iter results tokens ===\n")
        print(f"hutoken result tokens: {ht_result} \ndecoded text hutoken: {hutoken.decode(ht_result)} \ndecode text from correct: {hutoken.decode(tt_result)}\n")
        print(f"tiktoken result tokens: {tt_result}\n")
        print(f"huggingface result tokens: {hf_result}\n")

        print("\n=== iter results speed ===\n")
        print(f"hutoken result: {ht_perf_result} bytes / s.")
        print(f"tiktoken result: {tt_perf_result} bytes / s.")
        print(f"transformers result: {hf_perf_result} bytes / s.")

        raise AssertionError("Tokenizer results are not equal. Check tokenizers.")


    return ht_perf_result, tt_perf_result, hf_perf_result


def benchmark_test(document: str, iter: int, thread_number: int): # permutation based

    num_bytes = len(str.encode(document))
    print(f"document char len: {len(document)}")
    print(f"document num bytes: {num_bytes}")

    count_tt, count_hf = 0, 0
    ht_results, tt_results, hf_results = [], [], []

    for _ in range(iter):
        ht_result, tt_result, hf_result = benchmark(document, num_bytes, thread_number)

        ht_results.append(ht_result)
        tt_results.append(tt_result)
        hf_results.append(hf_result)

        if ht_result < tt_result: count_tt += 1
        elif ht_result < hf_result: count_hf += 1

    print("\n=== sig results ===\n")
    print(f"{'Library':<15}{'Throughput (bytes/s)':>25}{'Throughput (MB/s)':>25}")
    print("-" * 65)

    print(f"{'hutoken':<15}{mean(ht_results):>25,.2f}{mean(ht_results)/1e6:>25,.2f}")
    print(f"{'tiktoken':<15}{mean(tt_results):>25,.2f}{mean(tt_results)/1e6:>25,.2f}")
    print(f"{'transformers':<15}{mean(hf_results):>25,.2f}{mean(hf_results)/1e6:>25,.2f}")
    print('\n')

    print(f"hutoken vs tiktoken: {count_tt / iter} p-value")
    print(f"hutoken vs transformers: {count_hf / iter} p-value")


def read_file(file_path: str) -> str:
    path = pathlib.Path(file_path)
    if not path.is_file():
        print("error: the file '%s' does not exist.", file_path)
        raise RuntimeError("file not found.")
    with path.open("r", encoding="utf-8") as file:
        content = file.read()
        return content


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, required=True, help="file path")
    parser.add_argument("--iter", type=int, default=1000, help="number of iterations to run the benchmark")
    parser.add_argument("--chunk-size", type=int, default=None, help="chunk size to test tokenizer on")
    parser.add_argument("--thread-number", type=int, default=1, help="number of threads to use")
    args = parser.parse_args()

    document = read_file(args.file_path)

    if args.chunk_size is None:
        args.chunk_size = len(document)

    doc = document[:args.chunk_size]

    benchmark_test(doc, args.iter, args.thread_number)