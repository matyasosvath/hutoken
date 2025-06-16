import timeit
import pytest
import tiktoken
from transformers import AutoTokenizer
import hutoken

hutoken.initialize("NYTK/PULI-LlumiX-32K")
hf_enc = AutoTokenizer.from_pretrained("NYTK/PULI-LlumiX-32K")

print("Hutoken:\n")
print(hutoken.encode("Gorcsev Iván, a Rangoon teherhajó matróza még huszonegy éves sem volt, midőn elnyerte a fizikai Nobel-díjat."))
print(hutoken.encode("hello world"))
print("Hugging face:\n")
print(hf_enc.encode("Gorcsev Iván, a Rangoon teherhajó matróza még huszonegy éves sem volt, midőn elnyerte a fizikai Nobel-díjat."))
print(hf_enc.encode("hello world"))