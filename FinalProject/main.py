import tfidf_summary
import en_core_web_sm
from glob import glob
import random as rand

rand.seed(0)
nlp = en_core_web_sm.load()

def main():
    # The Corpus (List of all documents) and the text to be summarized (single document from corpus currently)
    #corpus = [open(x, encoding='windows-1252').read() for x in glob('data/raw_articles/*/*.txt')]
    corpus_path = glob('data/raw_articles/business/*.txt')
    corpus_path.sort()
    corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]

    # Randomly shuffle the corpus (With given seed for consistency)
    #rand.shuffle(corpus)

    # Grab the text from the corpus and create a spacy Doc object.
    #text = corpus[rand.randint(0, len(corpus))]
    title, text = corpus[0].split('\n', maxsplit=1)
    text = text.replace('\n', '')
    doc = nlp(text)

    # Select the 6 highest scores as the selected summarization sentences.
    # Scoring method is tf_idf score divided by the number of words in the sentence.
    text = ""
    for sentence in tfidf_summary.score_sentences(corpus, doc, 6):
        text += sentence + " "
    
    print(title + "\n---")
    print(text[:-1])

if __name__ == "__main__":
    main()