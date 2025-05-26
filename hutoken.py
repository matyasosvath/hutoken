import os
import sys
import traceback
from transformers import AutoTokenizer

try:
    import _hutoken
except ImportError:
    _hutoken = None

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

            if hasattr(hf_tokenizer, "vocab"):
                default = os.path.join(os.path.expanduser("~"), ".cache")
                cache_dir = os.getenv("XDG_CACHE_HOME", default)
                org_name, model_name = model_or_path.split("/")
                vocab_dir = os.path.join(cache_dir, f"hutoken/{org_name}")
                os.makedirs(vocab_dir, exist_ok=True)
                vocab_file = os.path.join(vocab_dir, f"{model_name}.txt")

                with open(vocab_file, "w", encoding="utf-8") as f:
                    for i, (token, idx) in enumerate(sorted(hf_tokenizer.vocab.items(), key=lambda x: x[1])):
                        try:
                            hex_parts = []
                            for b in token.encode('utf-8'):
                                hex_parts.append(f'0x{b:02X}')
                            hex_token = ''.join(hex_parts)

                            line = f"{hex_token} == {idx}\n"
                            f.write(line)
                        except Exception as e:
                            traceback.print_exc(file=sys.stderr)
            else:
                raise RuntimeError("hutoken: Could not extract vocab from Hugging Face tokenizer (no .vocab attribute).")

            try:
                result = _hutoken.initialize(vocab_file, *args, **kwargs)
                return result
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                raise RuntimeError(f"hutoken: Could not load Hugging Face tokenizer '{model_or_path}': {e}")
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"hutoken: Could not load Hugging Face tokenizer '{model_or_path}': {e}")

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
