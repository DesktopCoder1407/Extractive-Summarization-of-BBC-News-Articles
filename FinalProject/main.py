import tfidf_summary
from glob import glob
import random as rand

rand.seed(0)

def main():
    # The Corpus (List of all documents) and the text to be summarized (single document from corpus currently)
    #corpus = [open(x, encoding='windows-1252').read() for x in glob('data/raw_articles/*/*.txt')]
    corpus_path = glob('data/raw_articles/business/*.txt')
    corpus_path.sort()
    gold_corpus_path = glob('data/summarized_articles/business/*.txt')
    gold_corpus_path.sort()
    corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]
    gold_corpus = [open(x, encoding='windows-1252').read() for x in gold_corpus_path]

    # Randomly shuffle the corpus (With given seed for consistency)
    #rand.shuffle(corpus)

    # Grab the title and text from the corpus.
    #text = corpus[rand.randint(0, len(corpus))]
    title, text = corpus[0].split('\n', maxsplit=1)
    # text = text.replace('\n', '')

    # Select the 6 highest scores as the selected summarization sentences.
    # Scoring method is tf_idf score divided by the number of words in the sentence.
    summarized_text = ''
    for sentence in tfidf_summary.score_sentences(corpus, text, 6):
        summarized_text += sentence + ' '
    
    print(title + "\n---")
    print(summarized_text[:-1])

if __name__ == "__main__":
    main()