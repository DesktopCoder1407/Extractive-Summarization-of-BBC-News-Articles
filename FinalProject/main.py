import tfidf_summary
import naive_bayes_summary
from glob import glob

def main():
    # The Corpus (List of all documents) and the text to be summarized (single document from corpus currently)
    #corpus = [open(x, encoding='windows-1252').read() for x in glob('data/raw_articles/*/*.txt')]
    corpus_path = glob('data/raw_articles/*/*.txt')
    gold_corpus_path = glob('data/summarized_articles/*/*.txt')

    corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]
    gold_corpus = [open(x, encoding='windows-1252').read() for x in gold_corpus_path]

    # TODO: Make a train-test split to have a training corpus (train naive_bayes/update weights for tfidf) and a testing corpus (get rogue score)
    # TODO: Setup ROGUE-1, ROGUE-2 (Maybe 3?)
    # TODO: Below is all output. Probably keep for neatness sake.
    
    # Grab the title and text from the corpus.
    title, text = corpus[0].split('\n', maxsplit=1)

    # Select the 6 highest scores as the selected summarization sentences.
    # Scoring method is tf_idf score divided by the number of words in the sentence.
    summarized_text = ''
    for sentence in tfidf_summary.score_sentences(corpus, text, 6):
        summarized_text += sentence + ' '
    
    print('TF-IDF:')
    print(title + "\n---")
    print(summarized_text[:-1])

    summarized_text = ''
    for sentence in naive_bayes_summary.score_sentences(corpus, gold_corpus, text, 6):
        summarized_text += sentence + ' '
    
    print('NAIVE BAYES:')
    print(title + "\n---")
    print(summarized_text[:-1])

if __name__ == "__main__":
    main()