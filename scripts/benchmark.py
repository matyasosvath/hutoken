from typing import Any, cast

import time
import pathlib
import argparse
from statistics import mean
from transformers import logging

logging.set_verbosity_warning() # disable unnecessaryunnecessary  warnings


__doc__ = """Measure tokenizers speed and performance for a given document."""


def benchmark(document, num_bytes):

    import hutoken

    hutoken.initialize_encode('./vocabs/gpt2-vocab.txt')
    hutoken.encode("bemelegítés")

    start = time.perf_counter_ns()
    ht_result = hutoken.encode(document)
    end = time.perf_counter_ns()

    ht_perf_result = num_bytes / (end - start) * 1e9


    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    tt_result = enc.encode(document)
    end = time.perf_counter_ns()

    tt_perf_result = num_bytes / (end - start) * 1e9


    import transformers

    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    hf_result = hf_enc(document)
    end = time.perf_counter_ns()

    hf_perf_result = num_bytes / (end - start) * 1e9

    # print(f"\n=== iter results tokens ===\n")
    # print(f"hutoken result tokens: {ht_result}\n")
    # print(f"tiktoken result tokens: {tt_result}\n")
    # print(f"huggingface result tokens: {hf_result["input_ids"]}\n")

    # assert ht_result == tt_result == hf_result["input_ids"], "Tokenizer results are not equal. Check tokenizers."

    # print(f"\n=== iter results speed ===\n")
    # print(f"hutoken result: {ht_perf_result} bytes / s.")
    # print(f"tiktoken result: {tt_perf_result} bytes / s.")
    # print(f"transformers result: {hf_perf_result} bytes / s.")

    return ht_perf_result, tt_perf_result, hf_perf_result


def benchmark_test(document: str, iter: int): # permutation based

    num_bytes = len(str.encode(document))
    print(f"document num bytes: {num_bytes}")

    count_tt, count_hf = 0, 0
    ht_results, tt_results, hf_results = [], [], []

    for _ in range(iter):
        ht_result, tt_result, hf_result = benchmark(document, num_bytes)

        ht_results.append(ht_result)
        tt_results.append(tt_result)
        hf_results.append(hf_result)

        if ht_result < mean(tt_results): count_tt += 1
        elif ht_result < mean(hf_results): count_hf += 1

    print(f"\n=== sig results ===\n")
    print(f"hutoken avg. result: {mean(ht_results)} bytes / s.")
    print(f"tiktoken avg. result: {mean(tt_results)} bytes / s.")
    print(f"transformers avg. result: {mean(hf_results)} bytes / s.")

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
    parser.add_argument("--chunk-size", type=int, default=1000, help="chunk size to test tokenizer on")
    args = parser.parse_args()

    document = read_file(args.file_path)

    benchmark_test(document[:args.chunk_size], args.iter)
