import os
import tempfile
import sys

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
        if not hasattr(_hutoken, "initialize"):
            raise RuntimeError("hutoken: '_hutoken' C extension does not provide an 'initialize' method.")

        try:
            result = _hutoken.initialize(model_or_path, *args, **kwargs)
            return result
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "hutoken: transformers library is not installed. "
                "Please install it with `pip install transformers` to use Hugging Face models."
            )
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(model_or_path)

            if hasattr(hf_tokenizer, "vocab"):
                vocab_file = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt")
              
                for i, (token, idx) in enumerate(sorted(hf_tokenizer.vocab.items(), key=lambda x: x[1])):
                    try:
                        hex_parts = []
                        for b in token.encode('utf-8'):
                            hex_parts.append(f'0x{b:02X}')
                        hex_token = ''.join(hex_parts)

                        line = f"{hex_token} == {idx}\n"
                        vocab_file.write(line)
                    except Exception as e:
                        import traceback
                        traceback.print_exc(file=sys.stderr)

                vocab_file.close()
                vocab_file = vocab_file.name
            else:
                raise RuntimeError("hutoken: Could not extract vocab from Hugging Face tokenizer (no .vocab attribute).")

            try:
                result = _hutoken.initialize(vocab_file, *args, **kwargs)
                return result
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                raise RuntimeError(f"hutoken: Could not load Hugging Face tokenizer '{model_or_path}': {e}")
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"hutoken: Could not load Hugging Face tokenizer '{model_or_path}': {e}")

def encode(text):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")

    if not hasattr(_hutoken, "encode"):
        raise RuntimeError("hutoken: No tokenizer initialized. Call hutoken.initialize() before encoding.")

    try:
        tokens = _hutoken.encode(text)
        return tokens
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error encoding text: {e}")

def decode(tokens):
    if _hutoken is None:
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")

    if not hasattr(_hutoken, "decode"):
        raise RuntimeError("hutoken: No tokenizer initialized. Call hutoken.initialize() before decoding.")

    try:
        text = _hutoken.decode(tokens)
        return text
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error decoding tokens: {e}")

def bpe_train(*args, **kwargs):
    if _hutoken is not None and hasattr(_hutoken, "bpe_train"):
        return _hutoken.bpe_train(*args, **kwargs)
    raise RuntimeError(
        "hutoken: Native C extension '_hutoken' is not installed or does not provide 'bpe_train'."
    )