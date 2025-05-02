# huToken

**huToken** is a high-performance, C-based tokenization library for Python (inspired by `tiktoken` and `tokenizers`). It provides efficient encoding and decoding of text using Byte Pair Encoding (BPE). Work in progress.

## 📦 Installation

```sh
git clone git@github.com:matyasosvath/hutoken.git
pip install .
```

## 🛠 Usage

### 1️⃣ Training a New BPE Model
Train a new vocabulary from a text corpus:

```python
import hutoken

hutoken.bpe_train("your_text_data_here", 5000, "vocab.txt")
```

This creates a `vocab.txt` file containing token mappings.

### 2️⃣ Initializing the Tokenizer

#### Using a Local Vocabulary File (C Extension)
Before encoding or decoding, initialize the tokenizer with the vocabulary file:

```python
hutoken.initialize("vocab.txt")
```

#### Using a Hugging Face Model
You can also initialize directly from a Hugging Face tokenizer/model name:

```python
hutoken.initialize("NYTK/PULI-LlumiX-32K")
```

> **Note:** This requires the `transformers` library. Install it with:
> ```sh
> pip install transformers
> ```

### 3️⃣ Encoding Text
Convert text into token IDs:

```python
tokens = hutoken.encode("hello world")
print(tokens)  # example output: [14, 9, 19, 19, 24, 0, 23, 14, 17, 19, 11]
```

### 4️⃣ Decoding Tokens
Convert token IDs back into text:

```python
text = hutoken.decode([14, 9, 19, 19, 24, 0, 23, 14, 17, 19, 11])
print(text)  # example output: "hello world"
```

## 🧪 Running Tests

To verify that everything is working correctly, run:

```sh
pip install pytest pytest-memray
pytest --memray .
```

## 🐛 Debugging

To enable debug logs, set the `DEBUG` environment variable to `1` before running your code:

```sh
DEBUG=1 python your_script.py
```

## 🤗 Hugging Face Integration

- You can use any tokenizer from the Hugging Face Hub by passing its model name to `hutoken.initialize()`.
- The same `encode` and `decode` API is used, regardless of backend.
- If the model name or path is not a file, `hutoken` will attempt to load it via Hugging Face.

**Example:**
```python
import hutoken

# Initialize with a Hugging Face model
hutoken.initialize("NYTK/PULI-LlumiX-32K")
tokens = hutoken.encode("Ez egy Hugging Face teszt.")
text = hutoken.decode(tokens)
print(tokens)
print(text)
```

> If the model cannot be found or loaded, a clear error message will be shown.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.
