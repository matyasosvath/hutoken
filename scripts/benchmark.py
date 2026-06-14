import math
from typing import Any, cast

import time
import pathlib
import argparse
from statistics import mean, stdev

import os
import psutil

import hutoken

import tiktoken
enc = tiktoken.get_encoding("gpt2")

import transformers
hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")


__doc__ = """Measure tokenizers speed and performance for a given document."""

def measure_cpu_and_time(fn):
    process = psutil.Process(os.getpid())

    start_cpu = process.cpu_times()
    start = time.perf_counter_ns()

    result = fn()

    end = time.perf_counter_ns()
    end_cpu = process.cpu_times()

    cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    wall = (end - start) / 1e9

    cpu_util = (cpu_time / (wall * os.cpu_count())) * 100 if wall > 0 else 0.0

    return result, (end - start), cpu_util


def stats(xs):
    m = mean(xs)
    if len(xs) > 1:
        s = stdev(xs)
        ci95 = 1.96 * s / math.sqrt(len(xs))
    else:
        s = 0.0
        ci95 = 0.0
    return m, s, ci95


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
    document_batches = split_document(document, thread_number) if thread_number > 1 else document

    def run_ht_enc():
        return hutoken.encode(document) if thread_number == 1 else hutoken.batch_encode(document_batches, num_threads=thread_number)

    def run_ht_dec(encoded):
        return hutoken.decode(encoded) if thread_number == 1 else hutoken.batch_decode(encoded, num_threads=thread_number)

    def run_tt_enc():
        return enc.encode(document) if thread_number == 1 else enc.encode_ordinary_batch(document_batches, num_threads=thread_number)

    def run_tt_dec(encoded):
        return enc.decode(encoded) if thread_number == 1 else enc.decode_batch(encoded)

    def run_hf_enc():
        return hf_enc(document)["input_ids"] if thread_number == 1 else hf_enc(document_batches)["input_ids"]

    def run_hf_dec(encoded):
        return hf_enc.decode(encoded) if thread_number == 1 else hf_enc.batch_decode(encoded)

    # warmup
    hutoken.encode("bemelegítés")

    ht_result, ht_enc_time, ht_enc_cpu = measure_cpu_and_time(run_ht_enc)
    _, ht_dec_time, ht_dec_cpu = measure_cpu_and_time(lambda: run_ht_dec(ht_result))

    enc.encode("bemelegítés")

    tt_result, tt_enc_time, tt_enc_cpu = measure_cpu_and_time(run_tt_enc)
    _, tt_dec_time, tt_dec_cpu = measure_cpu_and_time(lambda: run_tt_dec(tt_result))

    hf_enc.model_max_length = int(1e30)  # silence!
    hf_enc.encode("bemelegítés")

    hf_result, hf_enc_time, hf_enc_cpu = measure_cpu_and_time(run_hf_enc)
    _, hf_dec_time, hf_dec_cpu = measure_cpu_and_time(lambda: run_hf_dec(hf_result))

    ht_enc_perf = num_bytes / ht_enc_time * 1e9
    ht_dec_perf = num_bytes / ht_dec_time * 1e9
    tt_enc_perf = num_bytes / tt_enc_time * 1e9
    tt_dec_perf = num_bytes / tt_dec_time * 1e9
    hf_enc_perf = num_bytes / hf_enc_time * 1e9
    hf_dec_perf = num_bytes / hf_dec_time * 1e9

    return (
        ht_enc_perf,
        tt_enc_perf,
        hf_enc_perf,
        ht_dec_perf,
        tt_dec_perf,
        hf_dec_perf,
        ht_enc_cpu,
        tt_enc_cpu,
        hf_enc_cpu,
        ht_dec_cpu,
        tt_dec_cpu,
        hf_dec_cpu,
    )


def benchmark_test(document: str, iter: int, thread_number: int):

    num_bytes = len(str.encode(document))
    print(f"document char len: {len(document)}")
    print(f"document num bytes: {num_bytes}")

    ht_enc_results, tt_enc_results, hf_enc_results = [], [], []
    ht_dec_results, tt_dec_results, hf_dec_results = [], [], []
    ht_enc_cpu_results, tt_enc_cpu_results, hf_enc_cpu_results = [], [], []
    ht_dec_cpu_results, tt_dec_cpu_results, hf_dec_cpu_results = [], [], []

    for _ in range(iter):
        (
            ht_enc,
            tt_enc,
            hf_enc,
            ht_dec,
            tt_dec,
            hf_dec,
            ht_enc_cpu,
            tt_enc_cpu,
            hf_enc_cpu,
            ht_dec_cpu,
            tt_dec_cpu,
            hf_dec_cpu,
        ) = benchmark(document, num_bytes, thread_number)

        ht_enc_results.append(ht_enc)
        tt_enc_results.append(tt_enc)
        hf_enc_results.append(hf_enc)
        ht_enc_cpu_results.append(ht_enc_cpu)
        tt_enc_cpu_results.append(tt_enc_cpu)
        hf_enc_cpu_results.append(hf_enc_cpu)

        ht_dec_results.append(ht_dec)
        tt_dec_results.append(tt_dec)
        hf_dec_results.append(hf_dec)
        ht_dec_cpu_results.append(ht_dec_cpu)
        tt_dec_cpu_results.append(tt_dec_cpu)
        hf_dec_cpu_results.append(hf_dec_cpu)

    def print_benchmark_summary(label: str, results: list[float], cpu_results: list[float]) -> None:
        m, s, ci = stats([x / 1e6 for x in results])
        cpu_m, _, _ = stats(cpu_results)
        print(f"{label:<15}{m:>15,.2f}{s:>15,.2f}{ci:>15,.2f}{cpu_m:>10.2f}")

    print("\n--- Encoding Performance ---")
    print(f"{'Library':<15}{'Mean (MB/s)':>15}{'Std (MB/s)':>15}{'95% CI':>15}{'CPU%':>10}")
    print("-" * 75)
    print_benchmark_summary('hutoken', ht_enc_results, ht_enc_cpu_results)
    print_benchmark_summary('tiktoken', tt_enc_results, tt_enc_cpu_results)
    print_benchmark_summary('transformers', hf_enc_results, hf_enc_cpu_results)

    print("\n--- Decoding Performance ---")
    print(f"{'Library':<15}{'Mean (MB/s)':>15}{'Std (MB/s)':>15}{'95% CI':>15}{'CPU%':>10}")
    print("-" * 75)
    print_benchmark_summary('hutoken', ht_dec_results, ht_dec_cpu_results)
    print_benchmark_summary('tiktoken', tt_dec_results, tt_dec_cpu_results)
    print_benchmark_summary('transformers', hf_dec_results, hf_dec_cpu_results)
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
    parser.add_argument("--tokenizer", type=str, default="openai-community/gpt2",
                        help="Tokenizer model or local vocab path")
    parser.add_argument("--special-file", type=str, default=None,
                        help="Special characters file for local vocab initialization")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Optional prefix string for initialization")
    parser.add_argument("--is-byte-encoder", action="store_true",
                        help="Enable byte encoder for hutoken initialization")
    parser.add_argument("--iter", type=int, default=1000,
                        help="number of iterations to run the benchmark")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="chunk size to test tokenizer on")
    parser.add_argument("--thread-number", type=int, default=1,
                        help="number of threads to use")
    parser.add_argument("--disable-arena", action="store_true",
                        help="Disable arena allocator in hutoken")
    parser.add_argument("--disable-pretokenizer", action="store_true",
                        help="Disable arena-based pretokenizer path in hutoken")
    parser.add_argument("--disable-aho-corasick", action="store_true",
                        help="Disable Aho-Corasick decode optimization in hutoken")
    parser.add_argument("--disable-bpe-optimized", action="store_true",
                        help="Disable optimized BPE merging in hutoken")
    args = parser.parse_args()

    document = read_file(args.file_path)

    if args.chunk_size is None:
        args.chunk_size = len(document)

    doc = document[:args.chunk_size]

    hutoken.initialize(
        args.tokenizer,
        args.special_file,
        prefix=args.prefix,
        is_byte_encoder=args.is_byte_encoder,
        use_arena=not args.disable_arena,
        use_pretokenizer=not args.disable_pretokenizer,
        use_aho_corasick=not args.disable_aho_corasick,
        use_bpe_optimized=not args.disable_bpe_optimized,
    )

    print("Hutoken ablation config:")
    print(f"  tokenizer: {args.tokenizer}")
    print(f"  disable_arena: {args.disable_arena}")
    print(f"  disable_pretokenizer: {args.disable_pretokenizer}")
    print(f"  disable_aho_corasick: {args.disable_aho_corasick}")
    print(f"  disable_bpe_optimized: {args.disable_bpe_optimized}")

    benchmark_test(doc, args.iter, args.thread_number)
