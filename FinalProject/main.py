import tfidf_summary
import naive_bayes_summary
import tokenizer
from glob import glob

def main():
    # The Corpus (List of all documents) and the text to be summarized (single document from corpus currently)
    #corpus = [open(x, encoding='windows-1252').read() for x in glob('data/raw_articles/*/*.txt')]
    corpus_path = glob('data/raw_articles/*/*.txt')
    gold_corpus_path = glob('data/summarized_articles/*/*.txt')

    corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]
    gold_corpus = [open(x, encoding='windows-1252').read() for x in gold_corpus_path]

    # TODO: Make a train-test split to have a training corpus (train naive_bayes/update weights for tfidf) and a testing corpus (get rogue score)
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

def rouge(summary: str, gold_summary: str, n: int):
    summary_tokens = tokenizer.tokenize(summary)
    gold_summary_tokens = tokenizer.tokenize(gold_summary)
    summary_n_grams = []
    gold_summary_n_grams = []

    # Create n_grams for summary
    for i in range(len(summary_tokens) - (n-1)):
        summary_n_grams.append(summary_tokens[i])
        for j in range(1, n):
            summary_n_grams[i] += ' ' + summary_tokens[i + j]
    
    # Create n_grams for gold_summary
    for i in range(len(gold_summary_tokens) - (n-1)):
        gold_summary_n_grams.append(gold_summary_tokens[i])
        for j in range(1, n):
            gold_summary_n_grams[i] += ' ' + gold_summary_tokens[i + j]
    
    # Find overlapping n_grams and calculate recall and precision
    overlapping = 0
    for gram in summary_n_grams:
        if gram in gold_summary_n_grams:
            overlapping += 1
    recall = overlapping / len(gold_summary_n_grams)
    precision = overlapping / len(summary_n_grams)

    return (recall, precision)

if __name__ == "__main__":
    main()