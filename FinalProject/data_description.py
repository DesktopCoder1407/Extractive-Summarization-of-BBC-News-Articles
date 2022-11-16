import tokenizer
from glob import glob

def get_corpus_split():
    corpus = []
    for subfolder in ['business', 'entertainment', 'politics', 'sport', 'tech']:
        corpus.append([open(x, encoding='windows-1252').read() for x in glob(f'data/raw_articles/{subfolder}/*.txt')])
    return corpus

# Get the Corpus
corpus = [open(x, encoding='windows-1252').read() for x in glob('data/raw_articles/*/*.txt')]
corpus_split = get_corpus_split()

# Sentences
print(f'Total sentences in the business category: {sum([len(tokenizer.sentencize(d)) for d in corpus_split[0]])}')
print(f'Total sentences in the entertainment category: {sum([len(tokenizer.sentencize(d)) for d in corpus_split[1]])}')
print(f'Total sentences in the politics category: {sum([len(tokenizer.sentencize(d)) for d in corpus_split[2]])}')
print(f'Total sentences in the sport category: {sum([len(tokenizer.sentencize(d)) for d in corpus_split[3]])}')
print(f'Total sentences in the tech category: {sum([len(tokenizer.sentencize(d)) for d in corpus_split[4]])}')
print(f'Total sentences in the entire corpus: {sum([len(tokenizer.sentencize(d)) for d in corpus])}\n')

# Tokens
print(f'Total tokens in the business category: {sum([len(tokenizer.tokenize(d)) for d in corpus_split[0]])}')
print(f'Total tokens in the entertainment category: {sum([len(tokenizer.tokenize(d)) for d in corpus_split[1]])}')
print(f'Total tokens in the politics category: {sum([len(tokenizer.tokenize(d)) for d in corpus_split[2]])}')
print(f'Total tokens in the sport category: {sum([len(tokenizer.tokenize(d)) for d in corpus_split[3]])}')
print(f'Total tokens in the tech category: {sum([len(tokenizer.tokenize(d)) for d in corpus_split[4]])}')
print(f'Total tokens in the entire corpus: {sum([len(tokenizer.tokenize(d)) for d in corpus])}\n')

# Vocabulary
print(f'Vocabulary for the business category: {sum([len(set(tokenizer.tokenize(d.lower()))) for d in corpus_split[0]])}')
print(f'Vocabulary for the entertainment category: {sum([len(set(tokenizer.tokenize(d.lower()))) for d in corpus_split[1]])}')
print(f'Vocabulary for the politics category: {sum([len(set(tokenizer.tokenize(d.lower()))) for d in corpus_split[2]])}')
print(f'Vocabulary for the sport category: {sum([len(set(tokenizer.tokenize(d.lower()))) for d in corpus_split[3]])}')
print(f'Vocabulary for the tech category: {sum([len(set(tokenizer.tokenize(d.lower()))) for d in corpus_split[4]])}')
print(f'Vocabulary for the entire corpus: {sum([len(set(tokenizer.tokenize(d.lower()))) for d in corpus])}')