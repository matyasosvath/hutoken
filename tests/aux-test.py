import timeit
import pytest
import tiktoken
from transformers import AutoTokenizer
import hutoken

hutoken.initialize("NYTK/PULI-LlumiX-32K")
#hf_enc = AutoTokenizer.from_pretrained("NYTK/PULI-LlumiX-32K")

print("Hutoken:\n")
print(hutoken.encode("Gorcsev Iván, a Rangoon teherhajó matróza még huszonegy éves sem volt, midőn elnyerte a fizikai Nobel-díjat."
    "Ilyen nagy jelentőségű tudományos jutalmat e poétikusan ifjú korban megszerezni példátlan nagyszerű teljesítmény,"
    "még akkor is, ha egyesek előtt talán szépséghibának tűnik majd, hogy Gorcsev Iván a fizikai Nobel-díjat "
    "a makao nevű kártyajátékon nyerte el, Noah Bertinus professzortól, akinek ezt a kitüntetést Stockholmban,"
    "néhány nappal előbb, a svéd király nyújtotta át, de végre is a kákán csomót keresők nem számítanak; "
    "a lényeg a fő: hogy Gorcsev Iván igenis huszonegy éves korában elnyerte a Nobel-díjat."))
print("Hugging face:\n")
print(hf_enc.encode("Gorcsev Iván, a Rangoon teherhajó matróza még huszonegy éves sem volt, midőn elnyerte a fizikai Nobel-díjat."
    "Ilyen nagy jelentőségű tudományos jutalmat e poétikusan ifjú korban megszerezni példátlan nagyszerű teljesítmény,"
    "még akkor is, ha egyesek előtt talán szépséghibának tűnik majd, hogy Gorcsev Iván a fizikai Nobel-díjat "
    "a makao nevű kártyajátékon nyerte el, Noah Bertinus professzortól, akinek ezt a kitüntetést Stockholmban,"
    "néhány nappal előbb, a svéd király nyújtotta át, de végre is a kákán csomót keresők nem számítanak; "
    "a lényeg a fő: hogy Gorcsev Iván igenis huszonegy éves korában elnyerte a Nobel-díjat."))
