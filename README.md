# huToken

**huToken** is a high-performance tokenization library for Python inspired by
`tokenizers` and `tiktoken`.

-   Provides efficient encoding and decoding of text using Byte Pair Encoding.
-   Strives to be compatible with every large language model with an open-source
    tokenizer.
-   Hugging Face integration is built-in, you can use any tokenizer from the
    Hugging Face Hub.
-   Training new vocabularies is possible, and the local vocabularies could be
    used just as easily as Hugging Face tokenizers.
-   Designed for both research and production use with an easy-to-use intuitive
    API.

If a bug is found please leave feedback with the exact details.

# Quick links

- [License](#license)
- [Performance](#performance)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Scripts](#scripts)
- [Data format](#data-format)
- [Contributing](#contributing)
- [Contact](#contact)

# License

This project is licensed under the [MIT License](LICENSE).

# Performance

Benchmarks are currently being performed, and we'll plan optimizations according
to the results.

# Requirements

-   Python >= 3.9
-   `transformers` (optional, for Hugging Face Hub integration).
-   Any other dependency the chosen tokenizer uses (e.g., `sentencepiece` and
    `protobuf` for Llama tokenizers).

    > If a tokenizer requires additional system packages, the library raises an
    > informative exception listing the missing dependency.

-   `emMorph` (optional, for morphological analyzer integration).

-   `foma` (optional, for morphological analyzer integration).

    > This library is dynamically linked to the C implementation of huToken.
    > Thus, this should not be installed using `pip`, but using your system's
    > package manager, or add the header files to `C:\Windows`, depending on
    > your OS.
    >
    > For example, in Debian and Ubuntu, you can use:
    > ```bash
    > apt install libfoma-dev
    > ```
    > In Arch Linux, you can install `foma` from AUR with your favourite
    > AUR helper.
    >
    > Make sure to install the header files alongside the foma binaries.

# Installation

## PyPI

This project is published to PyPI, if you want to use local vocabularies, just
run:

```bash
pip install hutoken
```

Or, if you want to use the Hugging Face integration:

```bash
pip install hutoken[transformers]
```

## From git repository

If you want to use the latest development version, you can install from source
using:

```bash
pip install git+https://github.com/matyasosvath/hutoken.git
```

## For development

If you want to contribute, setting up the library for development is easy:

1.  Clone the repository.

    ```bash
    git clone git@github.com/matyasosvath/hutoken.git
    cd hutoken
    ```

2.  Create a virtual environment (optional but recommended).

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate # or, on Windows: .\.venv\Scripts\activate
    ```

3.  Install the project

    We recommend installing in editable mode.

    ```bash
    pip install -e .
    ```

# Quickstart

```python
# Quickstart
import hutoken

# initialize from a local vocab file or a Hugging Face id
hutoken.initialize("gpt2_vocab.txt",
                   "gpt2_special_chars.txt",
                   is_byte_encoder=True)

# encode / decode
encoded = hutoken.encode("hello world")
print("tokens:", encoded)

decoded = hutoken.decode(encoded)
print("text:", decoded)
```

# Usage

Here we'll represent usage scenarios. Note that in all the examples, the
`hutoken` library is expected to be imported.

## Training a new BPE model

```python
hutoken.bpe_train("your_text_data_here", 5000, "vocab.txt")
```

This creates a `vocab.txt` file containing token mappings.

## Using a pre-trained tokenizer

### Local vocabulary file

```python
hutoken.initialize("vocab.txt")
```

If the tokenizer uses byte encoding, you should set `is_byte_encoder=True`
parameter as well. If there are any special character mappings, those could be
represented in a `special_chars.txt` file, which could also be passed to the
function as a second parameter.

For example, using GPT-2 tokenizer is done as follows:

```python
hutoken.initialize("gpt2_vocab.txt",
                   "gpt2_special_chars.txt",
                   is_byte_encoder=True)
```

### Using a Hugging Face model

```python
hutoken.initialize("NYTK/PULI-LlumiX-32K")
```

This requires the `transformers` dependency, and if the tokenizer uses any
additional dependencies, those should be installed as well.

## Encoding text

The `initialize` function should be called before encoding any text.

```python
tokens = hutoken.encode("hello world")
print(tokens) # example output: [14, 9, 19, 19, 24, 0, 23, 14, 17, 19, 11]
```

## Decoding tokens

Again, the `initialize` function should be called before decoding any tokens.

```python
text = hutoken.decode([14, 9, 19, 19, 24, 0, 23, 14, 17, 19, 11])
print(text) # example output: "hello world"
```

## Using multiple threads

During encoding or decoding you can use multiple threads.
The `initialize` function should be called here also.

```python
tokens = hutoken.batch_encode(["hello", " world"], num_threads = 4)
print(tokens) # example output: [[14, 9, 19, 19, 24], [0, 23, 24, 17, 19, 11]]
text = hutoken.batch_decode([[14, 9, 19, 19, 24], [0, 23, 24, 17, 19, 11]], num_threads = 4)
print(text) # example output: "hello world"
```

## Morphological analyzer

### Looking up a word's morphemes

```python
emmorph = hutoken.initialize_foma()
morphemes = hutoken.look_up_word(emmorph, "fejetlenséget")
print(morphemes)
# example output: [['fejetlenség', 'et'],
#                  ['fejetlen', 'ség', 'et'],
#                  ['fej', 'etlen', 'ség', 'et'],
#                  ['fej', 'etlen', 'ség', 'et']]

```

This requires the `emMorph` and `foma` dependencies.

Other look-up trees could be used, by initializing them with `foma`, and passing
the handle to `look_up_word`.

### Tokenizer integration

Work in progress.

## Debugging

To enable debug logs, set the `DEBUG` environment variable to `1` before running
your code:

```bash
DEBUG=1 python your_script.py
```

# Scripts

This library provides additional scripts. There are two which could be useful
for your use case.

## Converting JSON format to huToken data format

The `convert.py` script converts a JSON vocabulary file to the data format used
by huToken.

## Building morphological analyzer tree

The `emmorph.sh` script builds the emMorph morphological analyzer's look-up
tree.

# Data format

This library uses a different format from Hugging Face for storing vocabularies.

Both of these are low-level details of the library, however, for advanced usage,
it is helpful to know how huToken works under the hood.

They are broken into two different files.

## Vocabulary files

```plaintext
0x27 == 6
0x28 == 7
0x29 == 8
0x2A == 9
0x2B == 10
0x2C == 11
0x2D == 12
0x2E == 13
0x2F == 14
0x30 == 15
0x31 == 16
0x32 == 17
0x33 == 18
0x34 == 19
0x35 == 20
0x36 == 21
0x37 == 22
0x38 == 23
```

Each line is one token, where the characters are encoded into a hex string, and
their token values are stored after `==`.

## Special character mappings

If a tokenizer uses special characters, they are stored in another file. This is
fully optional, and is only used if the tokenizer uses this special mapping.

```
14 == <0x0E>
15 == <0x0F>
16 == <0x10>
17 == <0x11>
18 == <0x12>
19 == <0x13>
20 == <0x14>
21 == <0x15>
22 == <0x16>
23 == <0x17>
24 == <0x18>
25 == <0x19>
26 == <0x1A>
27 == <0x1B>
28 == <0x1C>
```

Each line is a character, where the ASCII codes are mapped to their replacement
characters.

# Contributing

Contributions are welcome! Please open an issue, or submit a pull request, which
follows these requirements:

-   Follow the conventional commits guideline.
-   Write unit tests for your code, either in C or Python, in the `tests`
    directory. Every single test should pass.
-   Format the code with `clang-format` using the `.clang-format` configuration
    file.
-   Fix every `clang-tidy` lint, using the `.clang-tidy` configuration file.
-   Document your changes either in code, or in the README.

# Contact

To personally contact the maintainers of the library, here are the core
developers working on the project:

- [Mátyás Osváth](https://github.com/matyasosvath)
- [Roland Gunics](https://github.com/gunicsroland)
- [Máté Norbert Molnár](https://github.com/matee8)
