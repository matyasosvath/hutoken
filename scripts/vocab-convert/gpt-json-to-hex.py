import json

token = 0

with open('vocab.json') as f:
    d = json.load(f)
    f = open('gpt2-vocab.txt', 'w')
	
    for s in d:
        hex = s.encode('utf8').hex()
        hex = hex.upper()
        format = '0x'.join(hex[i:i+2] for i in range(0, len(hex), 2))
        format = ''.join(['0x', format])
        f.write(format + ' == ' + str(token) + '\n')
        token = token + 1

f.close()