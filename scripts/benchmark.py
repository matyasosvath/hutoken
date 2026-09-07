import math
from itertools import zip_longest
from typing import Any, cast

import time
import pathlib
import argparse
from statistics import mean, stdev

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


def standard_deviation(results: list[float]) -> float:
    return stdev(results) if len(results) > 1 else 0.0


def print_performance_results(results: list[tuple[str, list[float]]]) -> None:
    print(
        f"{'Library':<15}{'Mean (bytes/s)':>20}{'Std Dev (bytes/s)':>20}"
        f"{'Mean (MB/s)':>20}{'Std Dev (MB/s)':>20}"
    )
    print("-" * 95)
    for library, measurements in results:
        average = mean(measurements)
        deviation = standard_deviation(measurements)
        print(
            f"{library:<15}{average:>20,.2f}{deviation:>20,.2f}"
            f"{average / 1e6:>20,.2f}{deviation / 1e6:>20,.2f}"
        )


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

    ht_enc_perf = num_bytes / (end - start) * 1e9

    hutoken.decode(ht_result) if thread_number == 1 else hutoken.batch_decode(ht_result, num_threads=thread_number)
    start = time.perf_counter_ns()
    hutoken.decode(ht_result) if thread_number == 1 else hutoken.batch_decode(ht_result, num_threads=thread_number)
    end = time.perf_counter_ns()
    ht_dec_perf = num_bytes / (end - start) * 1e9


    enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    tt_result = enc.encode(document) if thread_number == 1 else enc.encode_ordinary_batch(document_batches, num_threads=thread_number)
    end = time.perf_counter_ns()
    tt_enc_perf = num_bytes / (end - start) * 1e9

    enc.decode(tt_result) if thread_number == 1 else enc.decode_batch(tt_result)
    start = time.perf_counter_ns()
    enc.decode(tt_result) if thread_number == 1 else enc.decode_batch(tt_result)
    end = time.perf_counter_ns()
    tt_dec_perf = num_bytes / (end - start) * 1e9

    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    hf_result = hf_enc(document)["input_ids"] if thread_number == 1 else hf_enc(document_batches)["input_ids"]
    end = time.perf_counter_ns()
    hf_enc_perf = num_bytes / (end - start) * 1e9

    hf_enc.decode(hf_result) if thread_number == 1 else hf_enc.batch_decode(hf_result)
    start = time.perf_counter_ns()
    hf_enc.decode(hf_result) if thread_number == 1 else hf_enc.batch_decode(hf_result)
    end = time.perf_counter_ns()
    hf_dec_perf = num_bytes / (end - start) * 1e9


    normalized_ht = ht_result if thread_number == 1 else flatten(ht_result)
    normalized_tt = tt_result if thread_number == 1 else flatten(tt_result)
    normalized_hf = hf_result if thread_number == 1 else flatten(hf_result)
    if not normalized_ht == normalized_tt == normalized_hf:
        first_difference = next(
            i for i, values in enumerate(
                zip_longest(normalized_ht, normalized_tt, normalized_hf)
            ) if len(set(values)) > 1
        )
        raise AssertionError(
            "Tokenizer results differ at token index "
            f"{first_difference}; lengths: hutoken={len(normalized_ht)}, "
            f"tiktoken={len(normalized_tt)}, transformers={len(normalized_hf)}."
        )

    return ht_enc_perf, tt_enc_perf, hf_enc_perf, ht_dec_perf, tt_dec_perf, hf_dec_perf


def benchmark_test(document: str, iter: int, thread_number: int):

    num_bytes = len(str.encode(document))
    print(f"document char len: {len(document)}")
    print(f"document num bytes: {num_bytes}")

    ht_enc_results, tt_enc_results, hf_enc_results = [], [], []
    ht_dec_results, tt_dec_results, hf_dec_results = [], [], []

    for _ in range(iter):
        ht_enc, tt_enc, hf_enc, ht_dec, tt_dec, hf_dec = benchmark(document, num_bytes, thread_number)

        ht_enc_results.append(ht_enc)
        tt_enc_results.append(tt_enc)
        hf_enc_results.append(hf_enc)

        ht_dec_results.append(ht_dec)
        tt_dec_results.append(tt_dec)
        hf_dec_results.append(hf_dec)

    print("\n--- Encoding Performance ---")
    print_performance_results(
        [
            ("hutoken", ht_enc_results),
            ("tiktoken", tt_enc_results),
            ("transformers", hf_enc_results),
        ]
    )

    print("\n--- Decoding Performance ---")
    print_performance_results(
        [
            ("hutoken", ht_dec_results),
            ("tiktoken", tt_dec_results),
            ("transformers", hf_dec_results),
        ]
    )
    print('\n')


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
