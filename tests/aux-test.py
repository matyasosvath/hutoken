import hutoken
import tiktoken

tt_enc = tiktoken.get_encoding("gpt2")
hutoken.initialize_encode('vocabs/gpt2-vocab.txt')
sentence1 = "How can the net amount of entropy of the universe be massively decreased?"
sentence2 = "What I cannot create, I do not understand."

paragraph1 = (
        "piacon, a pénzváltók asztalainál - ahol sokatok hallgatott már engem - vagy másutt szoktam beszélni, ne csodálkozzatok,"
)

# print('Hutoken: ')
# print(hutoken.encode(sentence1))
# print('Tiktoken: ')
# print(tt_enc.encode(sentence1))
# print('Hutoken: ')
# print(hutoken.encode(sentence2))
# print('Tiktoken: ')
# print(tt_enc.encode(sentence2))
# print('Hutoken: ')
# print(hutoken.encode(paragraph1))
# print('Tiktoken: ')
# print(tt_enc.encode(paragraph1))
# problem az í űX őX ŐX ÖX ÜX ÓX ÚX ÉX ÁX ŰX ÍX
print('Hutoken: ')
print(hutoken.encode(paragraph1))
print('Tiktoken: ')
print(tt_enc.encode(paragraph1))
# print('Hutoken: ')
# print(hutoken.encode(sentence2))
# print('Tiktoken: ')
# print(tt_enc.encode(sentence2))



# print('Tiktoken: ')
# print(tt_enc.encode(', I'))
# print('Hutoken: ')
# print(hutoken.encode(', I'))
