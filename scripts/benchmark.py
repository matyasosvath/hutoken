from typing import Any, cast

import time
import pathlib
import argparse


__doc__ = """Measure tokenizers speed and performance for a given document."""


def benchmark(document: str) -> None:

    num_bytes = len(str.encode(document))
    print(f"num_bytes: {num_bytes}")

    import hutoken

    hutoken.initialize_encode('/home/osvathm/config/vocab.txt')
    hutoken.encode("bemelegítés")

    start = time.perf_counter_ns()
    ht_result = hutoken.encode(document)
    end = time.perf_counter_ns()
    print(f"hutoken \t{num_bytes / (end - start) * 1e9} bytes / s.")


    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    tt_result = enc.encode(document)
    end = time.perf_counter_ns()
    print(f"tiktoken \t{num_bytes / (end - start) * 1e9} bytes / s.")


    import transformers

    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    hf_result = hf_enc(document)
    end = time.perf_counter_ns()
    print(f"huggingface \t{num_bytes / (end - start) * 1e9} bytes / s.")

    print(f"\n=== results ===\n")
    print(f"hutoken result: {ht_result}.\n")
    print(f"tiktoken result: {tt_result}.\n")
    print(f"huggingface result: {hf_result}.\n")


def benchmark_significance_test(document: str) -> None: # permutation based
    pass



def read_file(file_path: str) -> str:
    path = pathlib.Path(file_path)
    if not path.is_file():
        print("Error: The file '%s' does not exist.", file_path)
        raise RuntimeError("File not found.")
    with path.open("r", encoding="utf-8") as file:
        content = file.read()
        return content


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='hugme cli tool')
    parser.add_argument("--file-path", type=str, required=True, help="file path")
    parser.add_argument("--chunk-size", type=int, nargs="+", default=[100,1000,10000], help="list of chunk size to test tokenizer on")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    document = read_file(args.file_path)

    for chunk_size in chunk_size:
        if len(document) < chunk_size:
            print("The document is shorter than the chunk size.")
            continue
        doc = document[:chunk_size]
        print(f"chunk size: {chunk_size}")
        print(f"document: {doc}")
        benchmark(doc)
