import timeit
import pytest
import tiktoken
from transformers import AutoTokenizer

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


def test_encode_raises_error():

    with pytest.raises(RuntimeError, match="Vocabulary is not initialized for encoding. Call 'initialize_encode' function first."):
        hutoken.encode("szia")


def test_decode_raises_error():

    with pytest.raises(RuntimeError, match="Vocabulary is not initialized for decoding. Call 'initialize_decode' function first."):
        hutoken.decode([1,2,3])


def test_encode_basic():

    hutoken.initialize("NYTK/PULI-LlumiX-32K")
    hf_enc = AutoTokenizer.from_pretrained("NYTK/PULI-LlumiX-32K")

    assert hutoken.encode(sentence1) == hf_enc.encode(sentence1)
    assert hutoken.encode(sentence2) == hf_enc.encode(sentence2)
    assert hutoken.encode(paragraph1) == hf_enc.encode(paragraph1)
    assert hutoken.encode(paragraph2) == hf_enc.encode(paragraph2)


def test_decode_basic():

    hutoken.initialize("NYTK/PULI-LlumiX-32K")

    assert hutoken.decode(hutoken.encode(sentence1)) == sentence1
    assert hutoken.decode(hutoken.encode(sentence2)) == sentence2
    assert hutoken.decode(hutoken.encode(paragraph1)) == paragraph1
    assert hutoken.decode(hutoken.encode(paragraph2)) == paragraph2


def test_encode_basic_with_tiktoken():

    tt_enc = tiktoken.get_encoding("gpt2")
    hutoken.initialize("openai-community/gpt2")

    assert hutoken.encode(sentence1) == tt_enc.encode(sentence1)
    assert hutoken.encode(sentence2) == tt_enc.encode(sentence2)
    assert hutoken.encode(paragraph1) == tt_enc.encode(paragraph1)
    assert hutoken.encode(paragraph2) == tt_enc.encode(paragraph2)


def test_decode_basic_with_tiktoken():

    hutoken.initialize("openai-community/gpt2")

    assert hutoken.decode(hutoken.encode(sentence1)) == sentence1
    assert hutoken.decode(hutoken.encode(sentence2)) == sentence2
    assert hutoken.decode(hutoken.encode(paragraph1)) == paragraph1
    assert hutoken.decode(hutoken.encode(paragraph2)) == paragraph2


@pytest.mark.benchmark(disable_gc=True)
def test_encode_speed():
    number = 10_000
    execution_time = timeit.timeit(
        lambda: hutoken.encode(sentence1),
        setup="import hutoken; hutoken.initialize('./vocabs/gpt2-vocab.txt', './vocabs/gpt2-vocab_special_chars.txt')",
        number=number,
    )
    print(f"Average execution time for {number} calls: {execution_time / number} seconds")

    assert execution_time / number < 1e-03, f"Average exectuion for function took too long: {execution_time / number}."


def test_decode_speed():
    number = 10_000
    execution_time = timeit.timeit(
        lambda: hutoken.decode(hutoken.encode(sentence1)),
        setup="import hutoken; hutoken.initialize('./vocabs/gpt2-vocab.txt', './vocabs/gpt2-vocab_special_chars.txt')",
        number=number
    )

    print(f"Average execution time for {number} calls: {execution_time / number} seconds")

    assert execution_time / number < 1e-03, f"Average exectuion for function took too long: {execution_time / number}."


def test_initialize_success():
    hutoken.initialize("NYTK/PULI-LlumiX-32K")


def test_initialize_invalid_format():
    with open('./vocabs/invalid-vocab.txt', 'w') as f:
        f.write("invalid_line_format\n")
    with pytest.raises(ValueError, match="Invalid format in vocab file."):
        hutoken.initialize('./vocabs/invalid-vocab.txt', './vocabs/invalid-vocab_special_chars.txt')


def test_decode_invalid_tokens():
    hutoken.initialize("openai-community/gpt2")

    invalid_tokens = [999999, -1, 50258]  # out of bounds and invalid tokens
    with pytest.raises(ValueError, match="Element must be non-negative and less than vocab size."):
        hutoken.decode(invalid_tokens)


def test_encode_with_hugginface():

    model_name = "NYTK/PULI-LlumiX-32K"
    text = "Ez egy Hugging Face teszt!"

    hutoken.initialize(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    ht_tokens = hutoken.encode(text)
    hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

    assert ht_tokens == hf_tokens, f"Encoded tokens differs: {ht_tokens} vs {hf_tokens}"


def test_decode_with_hugginface():

    model_name = "NYTK/PULI-LlumiX-32K"
    text = "Ez egy Hugging Face teszt!"

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
    hf_decoded = hf_tokenizer.decode(hf_tokens)

    hutoken.initialize(model_name)
    ht_decoded = hutoken.decode(hf_tokens)

    assert ht_decoded == hf_decoded, f"Decoded text differs: {ht_decoded} vs {hf_decoded}"

def test_decode_with_hugginface_using_hutoken_encdoe():

    model_name = "NYTK/PULI-LlumiX-32K"
    text = "Ez egy Hugging Face teszt!"

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
    hf_decoded = hf_tokenizer.decode(hf_tokens)

    hutoken.initialize(model_name)
    ht_tokens = hutoken.encode(text)
    ht_decoded = hutoken.decode(ht_tokens)

    assert ht_decoded == hf_decoded, f"Decoded text differs: {ht_decoded} vs {hf_decoded}"

def test_morphological_analyzer():
    handle = hutoken.initialize_foma()
    word = "fejetlenséget"

    expected = [['fejetlenség', 'et'],
                ['fejetlen', 'ség', 'et'],
                ['fej', 'etlen', 'ség', 'et'],
                ['fej', 'etlen', 'ség', 'et']]
    result = hutoken.look_up_word(handle, word)
    assert result == expected, f"Result array differs: {expected} vs {result}"
    
def test_morphological_analyzer_empty_word():
    handle = hutoken.initialize_foma()
    word = ""

    expected = []
    result = hutoken.look_up_word(handle, word)
    assert result == expected, f"Result array differs: {expected} vs {result}"
    
def test_morphological_analyzer_only_longest():
    handle = hutoken.initialize_foma()
    word = "fejetlenséget"

    expected = [['fej', 'etlen', 'ség', 'et']]
    result = hutoken.look_up_word(handle, word, True)
    assert result == expected, f"Result array differs: {expected} vs {result}"
