import json
import os
import sys
import traceback
try:
    from transformers import AutoTokenizer
except ImportError:
    transformers = None

try:
    import _hutoken
except ImportError:
    _hutoken = None

# the characters which GPT2Tokenizer encodes differently.
_SPECIAL_CHARS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
    147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 173
]


def _byte_to_unicode():
    visible_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    byte_values = list(visible_bytes)
    codepoints = list(visible_bytes)
    extra_codepoint = 256
    for byte in range(256):
        if byte not in visible_bytes:
            byte_values.append(byte)
            codepoints.append(extra_codepoint)
            extra_codepoint += 1
    return dict(zip(byte_values, map(chr, codepoints)))


def _uses_byte_level(tokenizer_json_path):
    try:
        with open(tokenizer_json_path, encoding="utf-8") as tokenizer_file:
            pre_tokenizer = json.load(tokenizer_file).get("pre_tokenizer", {})
    except (OSError, ValueError):
        return False

    def contains_byte_level(value):
        if isinstance(value, dict):
            return value.get("type") == "ByteLevel" or any(
                contains_byte_level(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(contains_byte_level(item) for item in value)
        return False

    return contains_byte_level(pre_tokenizer)


def _write_merges_file(tokenizer_json_path, merges_file_path):
    try:
        with open(tokenizer_json_path, encoding="utf-8") as tokenizer_file:
            model = json.load(tokenizer_file).get("model", {})
    except (OSError, ValueError):
        return False

    merges = model.get("merges") if model.get("type") == "BPE" else None
    if not merges:
        return False

    formatted_merges = []
    for merge in merges:
        if isinstance(merge, str):
            formatted_merges.append(merge)
        elif (
            isinstance(merge, list)
            and len(merge) == 2
            and all(isinstance(token, str) for token in merge)
        ):
            formatted_merges.append(" ".join(merge))
        else:
            return False

    try:
        with open(merges_file_path, "w", encoding="utf-8") as merges_file:
            merges_file.write("#version: 0.2\n")
            for merge in formatted_merges:
                merges_file.write(f"{merge}\n")
    except OSError:
        return False

    return True

def initialize(model_or_path, *args, **kwargs):
    """
    Initialize hutoken with either a vocab file path or a Hugging Face model name.
    """
    if os.path.isfile(model_or_path):
        if _hutoken is None:
            raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
        special_chars_file = args[0] if args else None
        merges_file = args[6] if len(args) > 6 else None
        if special_chars_file and not os.path.isfile(special_chars_file):
            raise ValueError(f"Special characters file '{special_chars_file}' does not exist.")

        if merges_file and not os.path.isfile(merges_file):
            raise ValueError(f"The provided merges file '{merges_file}' does not exist.")

        prefix = kwargs.get('prefix', None)
        is_byte_encoder = kwargs.get('is_byte_encoder', False)
        token_id = kwargs.get('token_id', -1)
        regex_pattern = kwargs.get('pattern', None)

        result = _hutoken.initialize(model_or_path, special_chars_file, prefix, is_byte_encoder, token_id, regex_pattern)
        return result
    else:
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(model_or_path)
        except OSError as e:
            raise ValueError("Could not download Hugging Face tokenizer "
                             f"'{model_or_path}': {e}")

        if not hasattr(hf_tokenizer, "vocab"):
            raise ValueError("Could not extract vocab from Hugging Face "
                             "tokenizer.")

        cache_dir = os.getenv("XDG_CACHE_HOME",
                              os.path.join(os.path.expanduser("~"), ".cache"))
        org_name, model_name = model_or_path.split("/")
        vocab_dir = os.path.join(cache_dir, f"hutoken/{org_name}/{model_name}")
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_file = os.path.join(vocab_dir, f"{model_name}.txt")

        hf_tokenizer.save_pretrained(vocab_dir)

        try:
            with open(vocab_file, "w", encoding="utf-8") as f:
                sorted_vocab = sorted(hf_tokenizer.vocab.items(),
                                      key=lambda item: item[1])
                for token, idx in sorted_vocab:
                    try:
                        hex_token = "".join(
                            f"0x{b:02X}" for b in token.encode("utf-8")
                        )
                        f.write(f"{hex_token} == {idx}\n")
                    except Exception as e:
                        sys.stderr.write(
                            f"Failed to process token '{token}': {e}"
                        )
        except IOError as e:
            traceback.print_exc(file=sys.stderr)
            raise IOError(f"Could not write vocab file to '{vocab_file}': {e}")

        hu_tokenized = hf_tokenizer.tokenize("hu")[0]
        prefix = hu_tokenized[0] if hu_tokenized != "hu" else None

        hf_kwargs = {"use_fast": False}
        if prefix is not None:
            hf_kwargs["add_prefix_space"] = False

        hf_tokenizer = AutoTokenizer.from_pretrained(model_or_path, **hf_kwargs)
        special_chars_file = os.path.join(vocab_dir, f"{model_name}_special_chars.txt")

        tokenizer_json_path = os.path.join(vocab_dir, "tokenizer.json")
        detected_byte_encoder = _uses_byte_level(tokenizer_json_path)
        is_byte_encoder = kwargs.pop(
            "is_byte_encoder", detected_byte_encoder
        )
        byte_encoder = _byte_to_unicode() if is_byte_encoder else None

        try:
            with open(special_chars_file, "w", encoding="utf-8") as f:
                for char in _SPECIAL_CHARS:
                    if byte_encoder is not None:
                        value = byte_encoder[char]
                    else:
                        value = ''.join(hf_tokenizer.tokenize(chr(char)))
                    if value == chr(char):
                        continue
                    f.write(f"{char} == {value}\n")
        except IOError as e:
            traceback.print_exc(file=sys.stderr)
            raise IOError("Could not write special characters file to "
                          f"'{special_chars_file}': {e}")


        merges_file_path = os.path.join(vocab_dir, "merges.txt")
        if (
            not os.path.isfile(merges_file_path)
            and not _write_merges_file(tokenizer_json_path, merges_file_path)
        ):
            merges_file_path = None
            sys.stderr.write(f"No merges.txt found for '{model_or_path}'. Continuing without merge rules.\n")

        try:
            result = _hutoken.initialize(
                vocab_file,
                special_chars_file,
                prefix,
                is_byte_encoder,
                merges_file_path=None,
                *args,
                **kwargs,
            )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError("An unexpected error occured during "
                               f"initialization: {e}") from e

        return result

def encode(text):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
    try:
        tokens = _hutoken.encode(text)
        return tokens
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error encoding text: {e}")

def batch_encode(texts, num_threads=1):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
    try:
        return _hutoken.batch_encode(texts, num_threads)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error encoding texts: {e}")

def decode(tokens):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
    try:
        text = _hutoken.decode(tokens)
        return text
    except ValueError as e:
        traceback.print_exc(file=sys.stderr)
        raise ValueError(f"hutoken: Error decoding tokens {tokens}: {e}")
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error decoding tokens: {e}")

def batch_decode(tokens, num_threads=1):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
    try:
        return _hutoken.batch_decode(tokens, num_threads)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error decoding tokens: {e}")

def bpe_train(*args, **kwargs):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or does not provide 'bpe_train'.")
    return _hutoken.bpe_train(*args, **kwargs)

def bbpe_train(*args, **kwargs):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or does not provide 'bbpe_train'.")
    return _hutoken.bbpe_train(*args, **kwargs)

def initialize_foma():
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed.")
    if not hasattr(_hutoken, "initialize_foma"):
        raise RuntimeError(
            "hutoken: '_hutoken' does not provide 'initialize_foma' "
            "or Foma support is not installed."
        )
    return _hutoken.initialize_foma()

def look_up_word(*args):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed.")
    if not hasattr(_hutoken, "look_up_word"):
        raise RuntimeError(
            "hutoken: '_hutoken' does not provide 'look_up_word' "
            "or Foma support is not installed."
        )
    return _hutoken.look_up_word(*args)
