import re

# Training text for the bigram model.
text = "the day was grey and bitter cold, and the dogs would not take the scent. the big black bitch had taken one " \
       "sniff at the bear tracks, backed off, and skulked back to the pack with her tail between her legs."

# Get all the tokens from the training text and find all the unique types (and number them).
tokens = re.findall(r'''\b\w+?\b|\.|,''', text)
types = {}
for token in tokens:
    if types.get(token) is None:
        types[token] = 1
    else:
        types[token] += 1

true_period = '\.' # Changes a "." to "\." to make it a literal period for regex.
bigrams = {}

# Get the bigrams. First dimension is the token we are looking for, second is the one we are given.
for gram in types:
    bigrams[gram] = {}
    for given_gram in types:
        # Store bigrams as a string fraction (just for displaying).
        bigrams[gram][given_gram] = f'''{len(re.findall(f'{given_gram if given_gram != "." else true_period} ?{gram if gram != "." else true_period}', text))}/{types[given_gram]}'''

# Begin the output text that will be written to the file, in the format "P({gram}|{given_gram}) = {fraction},  "
output_text = ''
for gram in types:
    for given_gram in types:
        output_text += f'P({gram}|{given_gram}) = {bigrams[gram][given_gram]},  '
    output_text += '\r\r'

# Write to file
file = open('output_bigram.txt', 'wt')
file.write(output_text)
file.close()
