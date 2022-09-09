import re
import random

# Dataset for the bi-gram model
dataset = "<s> the day was grey and bitter cold, and the dogs would not take the scent. the big black bitch had " \
          "taken one sniff at the bear tracks, backed off, and skulked back to the pack with her tail between her " \
          "legs. </s>"

# Get all the tokens from the training dataset and find all the unique types (and number them).
tokens = re.findall(r'''</?\w+?>|\b\w+?\b|\.|,''', dataset)
types = {}
for token in tokens:
    if types.get(token) is None:
        types[token] = 1
    else:
        types[token] += 1

# Preprocess the dataset
clean_dataset = ""
for token in tokens:
    clean_dataset += f"{token} "
clean_dataset = clean_dataset[:-1]

re_period = '\.'
bigrams = {}

# Get the bi-grams. First dimension is the token we are looking for, second is the one we are given.
for gram in types:
    bigrams[gram] = {}
    for given_gram in types:
        # Chance of a bi-gram showing
        bigrams[gram][given_gram] = len(re.findall(
            fr'(?:\s|^){given_gram if given_gram != "." else re_period} ?{gram if gram != "." else re_period}\s', clean_dataset))/types[given_gram]


def get_next_token(given_gram):
    possible_next = {}
    for gram in types:
        p = bigrams[gram][given_gram]
        if p != 0:
            possible_next[gram] = p

    # Sort the tokens by probability
    possible_next = dict(sorted(possible_next.items(), key=lambda item: item[1]))

    # Randomly select from the maximum probabilities
    max_p = list(possible_next.values())[0]
    pool = list()
    for token in possible_next:
        if possible_next[token] < max_p:
            break
        else:
            pool.append(token)
    return pool[random.randrange(0, len(pool))]


# Begin generating text using the bi-gram model.
generated_text = "<s>"

i = 0
while generated_text.split(" ")[-1] != "." and i < 100:
    generated_text += f' {get_next_token(generated_text.split(" ")[-1])}'
    i += 1
generated_text += " </s>"

print(generated_text)
