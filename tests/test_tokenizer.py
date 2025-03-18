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
    "ez így rosszabb lesz, de lehet, hogy jobb; ám csak azt kell vizsgálnotok, arra kell fordítanotok figyelmeteket, "
    "vajon igazat mondok-e vagy nem: a bírónak ugyanis ez az erénye, a szónoké pedig az igazmondás."
)

@pytest.fixture(autouse=False)
def setup():

    hutoken.initialize_encode('./vocabs/gpt2-vocab.txt')
    # hutoken.initialize_decode('./vocabs/gpt2-vocab.txt', 50256)

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
    assert hutoken.encode(sentence2) == tt_enc.encode(sentence2)
    assert hutoken.encode(paragraph1) == tt_enc.encode(paragraph1)
    assert hutoken.encode(paragraph2) == tt_enc.encode(paragraph2)


def test_decode_basic(setup):

    assert hutoken.decode(hutoken.encode(sentence1)) == sentence1
    assert hutoken.decode(hutoken.encode(sentence2)) == sentence2
    assert hutoken.decode(hutoken.encode(paragraph1)) == paragraph1
    assert hutoken.decode(hutoken.encode(paragraph2)) == paragraph2


@pytest.mark.benchmark(disable_gc=True)
def test_encode_speed():

    number = 10_000
    execution_time = timeit.timeit(
        f'hutoken.encode("{paragraph1}")',
        setup="import hutoken; hutoken.initialize_encode('./vocabs/gpt2-vocab.txt')",
        number=number
    )
    print(f"Average execution time for {number} calls: {execution_time / number} seconds")

    assert execution_time / number < 1e-03, f"Average exectuion for function took too long: {execution_time / number}."


def test_decode_speed():

    number = 10_000
    execution_time = timeit.timeit(
        f"hutoken.decode(hutoken.encode('{paragraph1}'))",
        setup="import hutoken; hutoken.initialize_decode(./vocabs/gpt2-vocab.txt', 50_256)",
        number=number
    )

    print(f"Average execution time for {number} calls: {execution_time / number} seconds")

    assert execution_time / number < 1e-03, f"Average exectuion for function took too long: {execution_time / number}."
