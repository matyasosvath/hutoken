from typing import Any, cast

import os
import time


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


if __name__ == "__main__":

    document = ""

    benchmark(document)
