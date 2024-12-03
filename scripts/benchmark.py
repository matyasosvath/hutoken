from typing import Any, cast

import os
import time


def benchmark(document: str) -> None:

    num_bytes = len(str.encode(document))
    print(f"num_bytes: {num_bytes}")

    import hutoken

    hutoken.initialize_encode('vocab', False)
    hutoken.encode("bemelegítés")

    start = time.perf_counter_ns()
    result = hutoken.encode(document)
    end = time.perf_counter_ns()
    print(f"hutoken \t{num_bytes / (end - start) * 1e9} bytes / s. result {result}")


    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    result = enc.encode(document)
    end = time.perf_counter_ns()
    print(f"tiktoken \t{num_bytes / (end - start) * 1e9} bytes / s. result {result}")


    import transformers

    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!
    hf_enc.encode("bemelegítés")

    start = time.perf_counter_ns()
    result = hf_enc(document)
    end = time.perf_counter_ns()
    print(f"huggingface \t{num_bytes / (end - start) * 1e9} bytes / s. result {result}")
