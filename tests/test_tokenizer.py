import gc
import timeit
import pytest
import hutoken


@pytest.fixture(autouse=True)
def disable_gc():
    gc.disable()
    gc.collect()
    yield
    gc.enable()


def test_encode_raises_error():

    with pytest.raises(RuntimeError, match="Vocabulary is not initialized for encoding. Call 'initialize_encode' function first."):
        hutoken.encode("szia")


def test_encode_basic():

    hutoken.initialize_encode('vocab', False)

    assert hutoken.encode("szia") == [55, 1]
    assert hutoken.encode("sziasztok") == [55, 56, 58]
    assert hutoken.encode("hello") == [14, 9, 19, 19, 24]
    assert hutoken.encode("hello hello") == [14, 9, 19, 19, 24, 0, 14, 9, 19, 19, 24]


@pytest.mark.benchmark(disable_gc=True)
def test_encode_benchmark_one_word_speed():

    number = 10_000
    execution_time = timeit.timeit(
        'hutoken.encode("sziasztok")',
        setup="import hutoken; hutoken.initialize_encode('vocab', False)",
        number=number
    )

    print(f"Average execution time for {number} calls: {execution_time / number} seconds")

    assert execution_time / number < 1e-05, f"Average exectuion for function took too long: {execution_time / number}."


def test_encode_benchmark_one_word_memory_usage():
    pass


def test_decode_raises_error():

    with pytest.raises(RuntimeError, match="Vocabulary is not initialized for decoding. Call 'initialize_decode' function first."):
        hutoken.decode([1,2,3])


def test_decode_basic():

    hutoken.initialize_decode('vocab', False)

    assert hutoken.decode([55, 1]) == "szia"
    assert hutoken.decode([55, 56, 58]) == "sziasztok"
    assert hutoken.decode([14, 9, 19, 19, 24]) == "hello"
    assert hutoken.decode([14, 9, 19, 19, 24, 0, 14, 9, 19, 19, 24]) == "hello hello"


def test_decode_benchmark_one_word_speed():

    number = 10_000
    execution_time = timeit.timeit(
        'hutoken.decode([14, 9, 19, 19, 24])',
        setup="import hutoken; hutoken.initialize_decode('vocab', False)",
        number=number
    )

    print(f"Average execution time for {number} calls: {execution_time / number} seconds")

    assert execution_time / number < 1e-05, f"Average exectuion for function took too long: {execution_time / number}."


def test_decode_benchmark_one_word_memory_usage():
    pass
