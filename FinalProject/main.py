import tfidf_summary
import naive_bayes_summary
import tokenizer
from glob import glob
import numpy
from sklearn.model_selection import train_test_split

def main():
    # The Corpus (List of all documents) and Gold Corpus (list of all reference summarized documents).
    corpus_path = glob('data/raw_articles/*/*.txt')
    gold_corpus_path = glob('data/summarized_articles/*/*.txt')
    corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]
    gold_corpus = [open(x, encoding='windows-1252').read() for x in gold_corpus_path]

    # Train-Test split: 90% testing, 10% training
    train_corpus, test_corpus, train_gold_corpus, test_gold_corpus = train_test_split(corpus, gold_corpus, test_size=0.1, random_state=0)

    # Summarized training data
    tfidf_summarized = tfidf_summary.score_corpus(corpus, test_corpus, 6)
    naive_bayes_summarized = naive_bayes_summary.score_corpus(train_corpus, train_gold_corpus, test_corpus, 6)
    
    # ROUGE-1 and ROUGE-2 scores for the TF-IDF Summarizer. Includes Recall, Precision, and F1-Score.
    tfidf_rouge = numpy.zeros((len(tfidf_summarized), 2, 2))
    for i in range(len(tfidf_summarized)):
        tfidf_rouge[i, :, 0] = rouge(tfidf_summarized[i], test_gold_corpus[i], 1)
        tfidf_rouge[i, :, 1] = rouge(tfidf_summarized[i], test_gold_corpus[i], 2)
    
    r_avg = numpy.average(tfidf_rouge, axis=0) # Average out the rouge data to get a single rouge score.
    print(f'TF-IDF ROUGE-1 Recall: {r_avg[0, 0]:.4f}')
    print(f'TF-IDF ROUGE-1 Precision: {r_avg[1, 0]:.4f}')
    print(f'TF-IDF ROUGE-1 F1-Score: {(2*r_avg[1, 0]*r_avg[0, 0])/(r_avg[1, 0]+r_avg[0, 0]):.4f}')
    print()
    print(f'TF-IDF ROUGE-2 Recall: {r_avg[0, 1]:.4f}')
    print(f'TF-IDF ROUGE-2 Precision: {r_avg[1, 1]:.4f}')
    print(f'TF-IDF ROUGE-2 F1-Score: {(2*r_avg[1, 1]*r_avg[0, 1])/(r_avg[1, 1]+r_avg[0, 1]):.4f}')
    print()

    # ROUGE-1 and ROUGE-2 scores for the Naive Bayes Summarizer. Includes Recall, Precision, and F1-Score.
    naive_bayes_rouge = numpy.zeros((len(naive_bayes_summarized), 2, 2))
    for i in range(len(naive_bayes_summarized)):
        naive_bayes_rouge[i, :, 0] = rouge(naive_bayes_summarized[i], test_gold_corpus[i], 1)
        naive_bayes_rouge[i, :, 1] = rouge(naive_bayes_summarized[i], test_gold_corpus[i], 2)
    
    r_avg = numpy.average(naive_bayes_rouge, axis=0) # Average out the rouge data to get a single rouge score.
    print(f'Naive Bayes ROUGE-1 Recall: {r_avg[0, 0]:.4f}')
    print(f'Naive Bayes ROUGE-1 Precision: {r_avg[1, 0]:.4f}')
    print(f'Naive Bayes ROUGE-1 F1-Score: {(2*r_avg[1, 0]*r_avg[0, 0])/(r_avg[1, 0]+r_avg[0, 0]):.4f}')
    print()
    print(f'Naive Bayes ROUGE-2 Recall: {r_avg[0, 1]:.4f}')
    print(f'Naive Bayes ROUGE-2 Precision: {r_avg[1, 1]:.4f}')
    print(f'Naive Bayes ROUGE-2 F1-Score: {(2*r_avg[1, 1]*r_avg[0, 1])/(r_avg[1, 1]+r_avg[0, 1]):.4f}')
    print()

    # TODO: Time TF-IDF vs Naive Bayes to display in paper.
    
    # An example summarization for each model.
    print("Example summaries grabbed from the test corpus:")

    # Grab the title and document from the corpus.
    title, doc = test_corpus[0].split('\n', maxsplit=1)

    # Select the 6 highest scores as the selected summarization sentences.
    summarized_text = ''
    for sentence in tfidf_summary.score_sentences(corpus, doc, 6):
        summarized_text += sentence + ' '
    
    print('TF-IDF:')
    print(title + "\n---")
    print(summarized_text[:-1])
    recall_1, precision_1 = rouge(summarized_text[:-1], test_gold_corpus[0], 1)
    print(f'\nROUGE-1 Recall: {recall_1:.4f}   ROUGE-1 Precision: {precision_1:.4f}')
    recall_2, precision_2 = rouge(summarized_text[:-1], test_gold_corpus[0], 2)
    print(f'ROUGE-2 Recall: {recall_2:.4f}   ROUGE-2 Precision: {precision_2:.4f}')

    summarized_text = ''
    for sentence in naive_bayes_summary.score_sentences(test_corpus, test_gold_corpus, doc, 6):
        summarized_text += sentence + ' '
    
    print('\nNAIVE BAYES:')
    print(title + "\n---")
    print(summarized_text[:-1])
    recall_1, precision_1 = rouge(summarized_text[:-1], test_gold_corpus[0], 1)
    print(f'\nROUGE-1 Recall: {recall_1:.4f}   ROUGE-1 Precision: {precision_1:.4f}')
    recall_2, precision_2 = rouge(summarized_text[:-1], test_gold_corpus[0], 2)
    print(f'ROUGE-2 Recall: {recall_2:.4f}   ROUGE-2 Precision: {precision_2:.4f}')

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
    for gram in set(summary_n_grams):
        if gram in gold_summary_n_grams:
            overlapping += 1
    recall = overlapping / len(gold_summary_n_grams)
    precision = overlapping / len(summary_n_grams)

    return (recall, precision)

if __name__ == "__main__":
    main()