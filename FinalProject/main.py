import tfidf_extractive_summary
import en_core_web_sm
from glob import glob
import random as rand

rand.seed(0)
nlp = en_core_web_sm.load()

def main():
    # The Corpus (List of all documents) and the text to be summarized (single document from corpus currently)
    corpus = [open(x, encoding='windows-1252').read() for x in glob('data/raw_articles/*/*.txt')]
    
    # Randomly shuffle the corpus (With given seed for consistency)
    rand.shuffle(corpus)

    # Grab the text from the corpus and create a spacy Doc object.
    text = corpus[rand.randint(0, len(corpus))]
    doc = nlp(text)

    # Select the 6 highest scores as the selected summarization sentences.
    # Scoring method is tf_idf score divided by the number of words in the sentence.
    print(tfidf_extractive_summary.score_sentences(corpus, doc, 6))

if __name__ == "__main__":
    main()