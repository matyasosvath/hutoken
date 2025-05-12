import os
import tempfile
import sys

try:
    import _hutoken
except ImportError:
    _hutoken = None

def debug_log(msg, *args):
    """Helper function for consistent debug logging"""
    print(f"DEBUG: {msg}", *args, file=sys.stderr)

def initialize(model_or_path, *args, **kwargs):
    """
    Initialize hutoken with either a vocab file path or a Hugging Face model name.
    """
    debug_log(f"Initialize called with model_or_path={model_or_path}, args={args}, kwargs={kwargs}")
    
    if os.path.isfile(model_or_path):
        debug_log(f"Using local file: {model_or_path}")
        if _hutoken is None:
            debug_log("_hutoken module not found")
            raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
        if not hasattr(_hutoken, "initialize"):
            debug_log("_hutoken module has no initialize attribute")
            raise RuntimeError("hutoken: '_hutoken' C extension does not provide an 'initialize' method.")
        
        debug_log(f"Calling _hutoken.initialize with file: {model_or_path}")
        try:
            result = _hutoken.initialize(model_or_path, *args, **kwargs)
            debug_log("_hutoken.initialize successful")
            return result
        except Exception as e:
            debug_log(f"Error in _hutoken.initialize: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    else:
        debug_log(f"Using Hugging Face model: {model_or_path}")
        try:
            debug_log("Importing AutoTokenizer")
            from transformers import AutoTokenizer
            debug_log("AutoTokenizer import successful")
        except ImportError as e:
            debug_log(f"Failed to import AutoTokenizer: {e}")
            raise ImportError(
                "hutoken: transformers library is not installed. "
                "Please install it with `pip install transformers` to use Hugging Face models."
            )
        try:
            debug_log(f"Loading tokenizer from {model_or_path}")
            hf_tokenizer = AutoTokenizer.from_pretrained(model_or_path)
            debug_log(f"Tokenizer type: {type(hf_tokenizer)}")
            debug_log(f"Tokenizer attributes: {dir(hf_tokenizer)}")
            
            # Try to extract vocab file path or dump vocab to a temp file
            if hasattr(hf_tokenizer, "vocab"):
                debug_log(f"Tokenizer has vocab attribute with {len(hf_tokenizer.vocab)} entries")
                # Create a temporary file with the right format
                vocab_file = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt")
                debug_log(f"Created temporary vocab file: {vocab_file.name}")
                
                # Sample a few tokens for debug
                sample_tokens = list(hf_tokenizer.vocab.items())[:5]
                debug_log(f"Sample tokens (first 5): {sample_tokens}")
                
                # Log special tokens if any
                if hasattr(hf_tokenizer, "special_tokens_map"):
                    debug_log(f"Special tokens: {hf_tokenizer.special_tokens_map}")
                
                for i, (token, idx) in enumerate(sorted(hf_tokenizer.vocab.items(), key=lambda x: x[1])):
                    # Convert each token to a clean hex representation
                    try:
                        # For debugging, capture the token type and its raw bytes
                        token_type = type(token)
                        raw_bytes = token.encode('utf-8')
                        
                        if i < 10 or i > len(hf_tokenizer.vocab) - 10:  # Log first and last 10
                            debug_log(f"Processing token #{i}: '{token}' (type={token_type}, bytes={raw_bytes}, idx={idx})")
                        
                        # The key is to match the format EXACTLY as expected by the C extension
                        # Convert the entire token to a single hex string with NO whitespace
                        hex_parts = []
                        for b in raw_bytes:
                            hex_parts.append(f'0x{b:02X}')
                        hex_token = ''.join(hex_parts)
                        
                        # Debug the hex representation
                        if i < 10 or i > len(hf_tokenizer.vocab) - 10:
                            debug_log(f"Token #{i} hex representation: {hex_token}")
                        
                        # Ensure there's exactly one space before and after '=='
                        line = f"{hex_token} == {idx}\n"
                        vocab_file.write(line)
                    except Exception as e:
                        debug_log(f"Error encoding token '{token}' (index {idx}): {e}")
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                
                vocab_file.close()
                debug_log(f"Finished writing vocab file: {vocab_file.name}")
                
                # Analyze the generated file for debugging
                debug_log("Validating generated vocab file...")
                try:
                    with open(vocab_file.name, 'r', encoding='utf-8') as f:
                        # Check first few lines
                        debug_log("First 10 lines of vocab file:")
                        for i, line in enumerate(f):
                            if i < 10:
                                debug_log(f"Line {i}: {line.strip()}")
                            else:
                                break
                        
                    # Also check last few lines
                    with open(vocab_file.name, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) > 10:
                            debug_log("Last 10 lines of vocab file:")
                            for i, line in enumerate(lines[-10:]):
                                debug_log(f"Line {len(lines)-10+i}: {line.strip()}")
                                
                    # Count lines to verify all tokens were written
                    with open(vocab_file.name, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                        debug_log(f"Vocab file has {line_count} lines (expected {len(hf_tokenizer.vocab)})")
                except Exception as e:
                    debug_log(f"Error validating vocab file: {e}")
                    
                vocab_file = vocab_file.name
                debug_log(f"Using vocab file path: {vocab_file}")
            else:
                debug_log("Tokenizer does not have a vocab attribute")
                debug_log(f"Available attributes: {dir(hf_tokenizer)}")
                raise RuntimeError("hutoken: Could not extract vocab from Hugging Face tokenizer (no .vocab attribute).")
            
            debug_log(f"Calling _hutoken.initialize with generated vocab file: {vocab_file}")
            try:
                result = _hutoken.initialize(vocab_file, *args, **kwargs)
                debug_log("_hutoken.initialize succeeded with generated vocab file")
                return result
            except Exception as e:
                debug_log(f"Error in _hutoken.initialize with generated vocab file: {e}")
                
                # Try to identify why the file format is invalid
                debug_log("Inspecting C extension's vocab file parsing...")
                try:
                    # Check if file exists and is readable
                    if not os.path.exists(vocab_file):
                        debug_log("ERROR: Vocab file does not exist!")
                    else:
                        debug_log(f"Vocab file exists with size: {os.path.getsize(vocab_file)} bytes")
                    
                    # Try to manually parse the file similar to how C might do it
                    with open(vocab_file, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i < 20:  # Check first 20 lines
                                line = line.strip()
                                parts = line.split(" == ")
                                if len(parts) != 2:
                                    debug_log(f"Invalid line format on line {i}: {line}")
                                    break
                                hex_str, value = parts
                                try:
                                    int_value = int(value)
                                    # Check if hex_str contains any whitespace
                                    if " " in hex_str:
                                        debug_log(f"Warning: Hex string contains whitespace on line {i}: {hex_str}")
                                except ValueError:
                                    debug_log(f"Invalid integer value on line {i}: {value}")
                                    break
                except Exception as debug_e:
                    debug_log(f"Error during manual file validation: {debug_e}")
                
                raise RuntimeError(f"hutoken: Could not load Hugging Face tokenizer '{model_or_path}': {e}")
        except Exception as e:
            debug_log(f"Uncaught exception in initialize: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"hutoken: Could not load Hugging Face tokenizer '{model_or_path}': {e}")

def encode(text):
    debug_log(f"encode called with text of length {len(text)}")
    debug_log(f"Text excerpt: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    if _hutoken is None:
        debug_log("_hutoken module not found")
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
        
    if not hasattr(_hutoken, "encode"):
        debug_log("_hutoken module has no encode attribute")
        raise RuntimeError("hutoken: No tokenizer initialized. Call hutoken.initialize() before encoding.")
    
    try:
        debug_log("Calling _hutoken.encode")
        tokens = _hutoken.encode(text)
        token_count = len(tokens)
        debug_log(f"Encoding successful: {token_count} tokens generated")
        
        # Show all tokens in debug output
        if token_count == 0:
            debug_log("No tokens generated")
        else:
            # For large token lists, split into multiple log lines
            if token_count > 100:
                debug_log(f"All {token_count} tokens (showing in chunks):")
                for i in range(0, token_count, 20):
                    chunk = tokens[i:i+20]
                    debug_log(f"  Tokens {i}-{i+len(chunk)-1}: {chunk}")
            else:
                debug_log(f"All {token_count} tokens: {tokens}")
                
        return tokens
    except Exception as e:
        debug_log(f"Error in _hutoken.encode: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error encoding text: {e}")

def decode(tokens):
    debug_log(f"decode called with {len(tokens)} tokens")
    
    if _hutoken is None:
        debug_log("_hutoken module not found")
        raise RuntimeError("hutoken: Native C extension '_hutoken' is not installed or failed to import.")
        
    if not hasattr(_hutoken, "decode"):
        debug_log("_hutoken module has no decode attribute")
        raise RuntimeError("hutoken: No tokenizer initialized. Call hutoken.initialize() before decoding.")
    
    # Show all input tokens in debug output
    token_count = len(tokens)
    if token_count == 0:
        debug_log("No tokens provided")
    else:
        # For large token lists, split into multiple log lines
        if token_count > 100:
            debug_log(f"All {token_count} input tokens (showing in chunks):")
            for i in range(0, token_count, 20):
                chunk = tokens[i:i+20]
                debug_log(f"  Tokens {i}-{i+len(chunk)-1}: {chunk}")
        else:
            debug_log(f"All {token_count} input tokens: {tokens}")
    
    try:
        debug_log("Calling _hutoken.decode")
        text = _hutoken.decode(tokens)
        debug_log(f"Decoding successful: {len(text)} characters generated")
        debug_log(f"Text excerpt: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        return text
    except Exception as e:
        debug_log(f"Error in _hutoken.decode: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"hutoken: Error decoding tokens: {e}")

def bpe_train(*args, **kwargs):
    if _hutoken is not None and hasattr(_hutoken, "bpe_train"):
        return _hutoken.bpe_train(*args, **kwargs)
    raise RuntimeError(
        "hutoken: Native C extension '_hutoken' is not installed or does not provide 'bpe_train'."
    )