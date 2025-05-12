from typing import Any, cast

import timeit
import pytest
import tiktoken
import transformers

import hutoken


sentence1 = "How can the net amount of entropy of the universe be massively decreased?"
sentence2 = "What I cannot create, I do not understand."
paragraph1 = (
    "Gorcsev Iván, a Rangoon teherhajó matróza még huszonegy éves sem volt, midőn elnyerte a fizikai Nobel-díjat."
    "Ilyen nagy jelentőségű tudományos jutalmat e poétikusan ifjú korban megszerezni példátlan nagyszerű teljesítmény,"
    "még akkor is, ha egyesek előtt talán szépséghibának tűnik majd, hogy Gorcsev Iván a fizikai Nobel-díjat "
    "a makao nevű kártyajátékon nyerte el, Noah Bertinus professzortól, akinek ezt a kitüntetést Stockholmban,"
    "néhány nappal előbb, a svéd király nyújtotta át, de végre is a kákán csomót keresők nem számítanak; "
    "a lényeg a fő: hogy Gorcsev Iván igenis huszonegy éves korában elnyerte a Nobel-díjat."
)
paragraph2 = (
    "Hogyan hatott rátok, athéni férfiak, vádlóim beszéde, nem tudom; én bizony magam is kis híján beléjük feledkeztem,"
    "olyan meggyőzően beszéltek. Ámbár igazat úgyszólván semmit nem mondtak. Sok hazugságuk közül leginkább egyet csodáltam:"
    "azt, amelyben kijelentették, hogy óvakodnotok kell, nehogy rászedjelek, minthogy félelmes szónok vagyok. Nem szégyellték,"
    "hogy nyomban tettel cáfolok rájuk, minthogy egyáltalán nem mutatkozom félelmes szónoknak; ez volt, gondolom, a legnagyobb"
    "szégyentelenségük; hacsak nem azt nevezik ők félelmes szónoknak, aki igazat mond. Mert ha igen, akkor elismerem:"
    "szónok vagyok, bár nem olyan, amilyennek ők elképzelik. Mert ők, mint mondom, alig mondtak valami igazat vagy éppen"
    "semmit sem; tőlem viszont a tiszta igazságot fogjátok hallani. De bizony, Zeuszra, athéni férfiak, nem ékes szavú,"
    "szólamokkal és cifra kifejezésekkel földíszített beszédeket, mint amilyeneket amazok mondanak, hanem az éppen kínálkozó"
    "szavakból álló, keresetlen beszédet. Mert hiszem, hogy mind igaz, amit mondok, és senki közületek tőlem mást ne várjon:"
    "mert hiszen nem is illenék, férfiak, hogy ebben az életkorban úgy lépjek elétek, mint valami beszédeket előre mintázgató ifjonc."
    "De éppen ezért kérve kérlek, athéni férfiak: ha azt hallanátok, hogy ugyanolyan szavakkal védekezem, mint amilyenekkel a"
    "piacon, a pénzváltók asztalainál - ahol sokatok hallgatott már engem - vagy másutt szoktam beszélni, ne csodálkozzatok,"
    "ne is zajongjatok miatta. Mert így áll a dolog: most állok először törvényszék előtt, több mint hetvenesztendős koromban,"
    "és szörnyen idegen számomra az itt szokásos beszédmód. Így hát, mint ahogy, ha történetesen valóban idegen volnék, megengednétek,"
    "hogy azon a nyelven és azon a módon szóljak, amelyen nevelkedtem, éppen úgy most is azzal a jogos, "
    "legalábbis nekem jogosnak tetsző kéréssel fordulok hozzátok, hogy adjatok engedélyt saját beszédmodoromra - lehet, hogy "
    "ez így rosszabb lesz, de lehet, hogy jobb; ám csak azt kell vizsgálnotok, arra kell fordítanótok figyelmeteket, "
    "vajon igazat mondok-e vagy nem: a bírónak ugyanis ez az erénye, a szónoké pedig az igazmondás."
)

@pytest.fixture(autouse=False)
def setup():

    hutoken.initialize('./vocabs/gpt2-vocab.txt')

    tt_enc = tiktoken.get_encoding("gpt2")

    hf_enc = cast(Any, transformers).GPT2TokenizerFast.from_pretrained("gpt2")
    hf_enc.model_max_length = 1e30  # silence!

    yield tt_enc, hf_enc


def test_encode_raises_error():

    with pytest.raises(RuntimeError, match="Vocabulary is not initialized for encoding. Call 'initialize_encode' function first."):
        hutoken.encode("szia")


def test_decode_raises_error():

    with pytest.raises(RuntimeError, match="Vocabulary is not initialized for decoding. Call 'initialize_decode' function first."):
        hutoken.decode([1,2,3])


def test_encode_basic(setup):

    tt_enc, hf_enc = setup

    assert hutoken.encode(sentence1) == tt_enc.encode(sentence1)
    # assert hutoken.encode(sentence2) == tt_enc.encode(sentence2)
    # assert hutoken.encode(paragraph1) == tt_enc.encode(paragraph1)
    # assert hutoken.encode(paragraph2) == tt_enc.encode(paragraph2)


def test_decode_basic(setup):

    assert hutoken.decode(hutoken.encode(sentence1)) == sentence1
    # assert hutoken.decode(hutoken.encode(sentence2)) == sentence2
    # assert hutoken.decode(hutoken.encode(paragraph1)) == paragraph1
    # assert hutoken.decode(hutoken.encode(paragraph2)) == paragraph2


@pytest.mark.benchmark(disable_gc=True)
def test_encode_speed():

    number = 10_000
    execution_time = timeit.timeit(
        f'hutoken.encode("{sentence1}")',
        setup="import hutoken; hutoken.initialize('./vocabs/gpt2-vocab.txt')",
        number=number
    )
    print(f"Average execution time for {number} calls: {execution_time / number} seconds")

    assert execution_time / number < 1e-03, f"Average exectuion for function took too long: {execution_time / number}."


def test_decode_speed():

    number = 10_000
    execution_time = timeit.timeit(
        f"hutoken.decode(hutoken.encode('{sentence1}'))",
        setup="import hutoken; hutoken.initialize('./vocabs/gpt2-vocab.txt')",
        number=number
    )

    print(f"Average execution time for {number} calls: {execution_time / number} seconds")

    assert execution_time / number < 1e-03, f"Average exectuion for function took too long: {execution_time / number}."


def test_initialize_success():
    """Test successful initialization of the tokenizer."""
    hutoken.initialize('./vocabs/gpt2-vocab.txt')


def test_initialize_invalid_format():
    """Test that initialize raises ValueError for an invalid vocab file format."""
    with open('./vocabs/invalid-vocab.txt', 'w') as f:
        f.write("invalid_line_format\n")
    with pytest.raises(ValueError, match="Vocab file is empty or contains no valid entries."):
        hutoken.initialize('./vocabs/invalid-vocab.txt')
        
        
def test_encode():
    """Test the encode function of hutoken."""
    hutoken.initialize('./vocabs/gpt2-vocab.txt')

    text = "Hello, world!"
    encoded_tokens = hutoken.encode(text)

    assert isinstance(encoded_tokens, list), "Encoded result should be a list"
    assert all(isinstance(token, int) for token in encoded_tokens), "All elements in the encoded result should be integers"

    print(f"Encoded tokens: {encoded_tokens}")
    

def test_decode_invalid_tokens():
    """Test that decode raises ValueError for invalid tokens."""
    hutoken.initialize('./vocabs/gpt2-vocab.txt')

    invalid_tokens = [999999, -1, 50258]  # Out of bounds or invalid tokens
    with pytest.raises(ValueError, match="Element must be non-negative and less than vocab size."):
        hutoken.decode(invalid_tokens)


def test_decode_using_tiktoken_encode():
    """Test decoding using tiktoken."""
    hutoken.initialize('./vocabs/gpt2-vocab.txt')
    tt_enc = tiktoken.get_encoding("gpt2")

    word = "entropy"
    encoded = tt_enc.encode(word)
    print(f"Encoded tokens (tiktoken): {encoded}")
    decoded = hutoken.decode(encoded)
    print(f"Decoded text (tiktoken): {decoded}")

    assert decoded == word, f"Decoded text does not match original. Decoded: {decoded}, Original: {word}"


def test_decode_whole_sentence_using_tiktoken_encode():
    """Test decoding a whole sentence encoded using tiktoken."""
    hutoken.initialize('./vocabs/gpt2-vocab.txt')
    tt_enc = tiktoken.get_encoding("gpt2")

    sentence = "How can the net amount of entropy of the universe be massively decreased?"
    encoded = tt_enc.encode(sentence)
    print(f"Encoded tokens (tiktoken): {encoded}")
    decoded = hutoken.decode(encoded)
    print(f"Decoded text (tiktoken): {decoded}")

    assert decoded == sentence, f"Decoded text does not match original. Decoded: {decoded}, Original: {sentence}"

def test_huggingface_initialize_and_encode_decode():
    """Test hutoken Hugging Face integration with a model name."""
    import pytest  # Import pytest here to avoid UnboundLocalError
    
    try:
        import transformers
    except ImportError:
        pytest.skip("transformers not installed")

    # Try a model we know has a vocab dictionary
    model_name = "NYTK/PULI-LlumiX-32K"
    
    # Include Unicode, special characters, and multi-byte sequences
    test_text = "Ez egy Hugging Face teszt! 😊 こんにちは [CLS] <|endoftext|>"
    
    try:
        # Check if model has a vocab attribute before testing
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if not hasattr(hf_tokenizer, "vocab"):
            pytest.skip(f"Model {model_name} doesn't have a .vocab attribute")
            
        # Initialize hutoken with a Hugging Face model
        hutoken.initialize(model_name)
        
        # Encode and decode using hutoken
        tokens = hutoken.encode(test_text)
        decoded = hutoken.decode(tokens)
        
        # Encode and decode using transformers directly
        hf_tokens = hf_tokenizer.encode(test_text, add_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(hf_tokens)
        
        # Check decoded text rather than token IDs
        assert decoded == hf_decoded, f"Decoded text differs: {decoded} vs {hf_decoded}"
        
        # Optional: Compare token counts, which should be same even if IDs differ
        assert len(tokens) == len(hf_tokens), f"Token count differs: {len(tokens)} vs {len(hf_tokens)}"
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")
