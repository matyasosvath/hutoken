import os
import sys
import traceback
from transformers import AutoTokenizer

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

def initialize(model_or_path, *args, **kwargs):
    """
    Initialize hutoken with either a vocab file path or a Hugging Face model name.
    """
    if os.path.isfile(model_or_path):
        if _hutoken is None:
            raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
        result = _hutoken.initialize(model_or_path, *args, **kwargs)
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
        vocab_dir = os.path.join(cache_dir, f"hutoken/{org_name}")
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_file = os.path.join(vocab_dir, f"{model_name}.txt")

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

        special_chars_file = os.path.join(vocab_dir, f"{model_name}_special_chars.txt")

        try:
            with open(special_chars_file, "w", encoding="utf-8") as f:
                for char in _SPECIAL_CHARS:
                    value = ''.join(hf_tokenizer.tokenize(chr(char)))
                    f.write(f"{char} == {value}\n")
        except IOError as e:
            traceback.print_exc(file=sys.stderr)
            raise IOError("Could not write special characters file to "
                          f"'{special_chars_file}': {e}")

        try:
            result = _hutoken.initialize(vocab_file, *args, **kwargs)
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

def decode(tokens):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
    try:
        text = _hutoken.decode(tokens)
        return text
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error decoding tokens: {e}")


def bpe_train(*args, **kwargs):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or does not provide 'bpe_train'.")
    return _hutoken.bpe_train(*args, **kwargs)

def initialize_foma():
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed.")
    if _hutoken.initialize_foma() is None:
        raise RuntimeError("hutoken: '_hutoken' does not provide 'initialize_foma'.")
    return _hutoken.initialize_foma()

def look_up_word(*args):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or does not provide 'look_up_word'.")
    return _hutoken.look_up_word(*args)
