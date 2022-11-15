import en_core_web_sm
from glob import glob
import matplotlib.pyplot as plt
import statistics
# Features: position of sentences in the document, sentence length, presence of uppercase words, similarity of the sentence to the document title, Uppercase Word Feature

# P(s in S | Features) = P(features | s in S) / P(features)
nlp = en_core_web_sm.load()

corpus_path = glob('data/raw_articles/business/*.txt')
corpus_path.sort()
corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]

title, text = corpus[0].split('\n', maxsplit=1)
# text = text.replace('\n', '')
doc = nlp(text)
doc2 = nlp(open('data/summarized_articles/business/001.txt', encoding='windows-1252').read())

for token in doc:
    print(token)


# Sentence Length: <20, 20-35, >35 [???]
# Paragraph Location: Beginning, Middle, End