import re

text = '<s> a a b c h d b f </s> <s> b c h h a d f f a h b </s> <s> b b a a h h c c h d d f </s> ' \
       '<s> a b f f h c c d f f h </s> <s> h h f c f c a a c c d d d </s>'

tokens = text.split()
types = ['<s>', 'a', 'b', 'c', 'd', 'f', 'h', '</s>']

bigrams = {}
for gram in types:
    bigrams[gram] = {}
    for given_gram in types:
        bigrams[gram][given_gram] = f'P({gram}|{given_gram}) = {len(re.findall(f"{given_gram} {gram}", text))}/{tokens.count(given_gram)}'

output_text = ''
for given_gram in types:
    for gram in types:
        output_text += f'{bigrams[gram][given_gram]},'
    output_text = output_text[:-1] + '\n'

file = open('output_bigram.txt', 'wt')
file.write(output_text)
file.close()
