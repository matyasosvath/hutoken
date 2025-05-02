import os

try:
    import _hutoken
except ImportError:
    _hutoken = None

_hf_tokenizer = None
_hf_tokenizer_name = None

def initialize(model_or_path, *args, **kwargs):
    """
    Initialize hutoken with either a vocab file path or a Hugging Face model name.
    """
    global _hf_tokenizer, _hf_tokenizer_name

    if os.path.isfile(model_or_path):
        _hf_tokenizer = None
        _hf_tokenizer_name = None
        if _hutoken is None:
            raise RuntimeError(
                "hutoken: Native C extension '_hutoken' is not installed or failed to import. "
                "Cannot initialize from vocab file '{}'."
                .format(model_or_path)
            )
        if not hasattr(_hutoken, "initialize"):
            raise RuntimeError(
                "hutoken: '_hutoken' C extension does not provide an 'initialize' method. "
                "Cannot initialize from vocab file '{}'."
                .format(model_or_path)
            )
        return _hutoken.initialize(model_or_path, *args, **kwargs)
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "hutoken: transformers library is not installed. "
                "Please install it with `pip install transformers` to use Hugging Face models."
            )
        try:
            _hf_tokenizer = AutoTokenizer.from_pretrained(model_or_path)
            _hf_tokenizer_name = model_or_path
        except Exception as e:
            raise RuntimeError(
                f"hutoken: Could not load Hugging Face tokenizer '{model_or_path}': {e}"
            )

def encode(text):
    if _hf_tokenizer is not None:
        return _hf_tokenizer.encode(text)
    if _hutoken is not None and hasattr(_hutoken, "encode"):
        return _hutoken.encode(text)
    raise RuntimeError(
        "hutoken: No tokenizer initialized. "
        "Call hutoken.initialize() with a valid vocab file path or Hugging Face model name before encoding."
    )

def decode(tokens):
    if _hf_tokenizer is not None:
        return _hf_tokenizer.decode(tokens)
    if _hutoken is not None and hasattr(_hutoken, "decode"):
        return _hutoken.decode(tokens)
    raise RuntimeError(
        "hutoken: No tokenizer initialized. "
        "Call hutoken.initialize() with a valid vocab file path or Hugging Face model name before decoding."
    )

def bpe_train(*args, **kwargs):
    if _hutoken is not None and hasattr(_hutoken, "bpe_train"):
        return _hutoken.bpe_train(*args, **kwargs)
    raise RuntimeError(
        "hutoken: Native C extension '_hutoken' is not installed or does not provide 'bpe_train'."
    )