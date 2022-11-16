# Possible Other Vector: Fixed Phrase: Sentence contains a phrase that is prespecified.
from glob import glob
import numpy
import tokenizer
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split

def main():
    corpus_path = glob('data/raw_articles/*/*.txt')
    corpus_y_path = glob('data/summarized_articles/*/*.txt')

    corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]
    gold_corpus = [open(x, encoding='windows-1252').read() for x in corpus_y_path]
    train_corpus, test_corpus, train_gold_corpus, test_gold_corpus = train_test_split(corpus, gold_corpus, test_size=0.1, random_state=0)

    classifier = CategoricalNB()

    training_vector = get_vector(train_corpus, train_gold_corpus)
    classifier.fit(training_vector[:, :-1], training_vector[:, -1])

    testing_vector = get_vector(test_corpus, test_gold_corpus)

    print('Mean training accuracy for the classifier: ')
    print(classifier.score(training_vector[:, :-1], training_vector[:, -1]))
    print('Mean testing accuracy for the classifier: ')
    print(classifier.score(testing_vector[:, :-1], testing_vector[:, -1]))

def score_sentences(corpus: list[str], gold_corpus: list[str], text: str, n: int = 0, classifier: CategoricalNB | None = None):
    chosen_sentences = []

    # Create the classifier and fit the corpus to the classifier.
    if classifier is None:
        classifier = CategoricalNB()
        training_vector = get_vector(corpus, gold_corpus)
        classifier.fit(training_vector[:, :-1], training_vector[:, -1])

    testing_vector = get_vector([text], [text])[:, :-1]
    if n == 0: # Select sentences chosen by the classifier.
        for sent, pred in zip(tokenizer.sentencize(text), classifier.predict(testing_vector)):
            if pred == 1:
                chosen_sentences.append(sent)
    else: # Select top n sentences.
        pred = list(zip(classifier.predict_log_proba(testing_vector)[:, 1], zip(range(len(tokenizer.sentencize(text))), tokenizer.sentencize(text))))
        pred.sort(reverse=True)
        pred = pred[:n]
        pred = [x[1] for x in pred]
        pred.sort()
        for i, sent in pred:
            chosen_sentences.append(sent)
    
    return chosen_sentences

def score_corpus(training_corpus: list[str], training_gold_corpus: list[str], testing_corpus: list[str], n: int = 0):
    summaries = []

    # Create the classifier and fit the corpus to the classifier.
    classifier = CategoricalNB()
    training_vector = get_vector(training_corpus, training_gold_corpus)
    classifier.fit(training_vector[:, :-1], training_vector[:, -1])
    
    # Summarize each document within the testing corpus.
    for doc in testing_corpus:
        doc_summary = ''
        for sentence in score_sentences(testing_corpus, testing_corpus, doc, n, classifier):
            doc_summary += sentence + ' '
        summaries.append(doc_summary[:-1])

    return summaries

def get_vector(corpus, gold_corpus):
    # Training Vector. Format: [SentenceLength, ParagraphLocation, SimilarityToTitle, UppercaseWords, CATEGORY]
    vector = []

    # Loop through each document and corresponding summarized document and add their vectors to the total vector.
    for doc, summarized_doc in zip(corpus, gold_corpus):
        title, doc = doc.split('\n', maxsplit=1)

        # Cache regex lists
        doc_sents = tokenizer.sentencize(doc)
        doc_paras = tokenizer.paragraphize(doc)
        doc_para_sents = [tokenizer.sentencize(para) for para in doc_paras]

        # Primary features of the document, stored in 2D list format.
        doc_features = []
        for sent in doc_sents:
            doc_features.append([
                get_sent_len_cat(sent),
                get_para_loc_cat(sent, doc_para_sents),
                get_title_similarity(sent, title),
                get_contains_uppercase(sent),
                get_category(sent, tokenizer.sentencize(summarized_doc))
            ])
        vector.extend(doc_features)
    
    return numpy.array(vector)

def get_sent_len_cat(sentence):
    # Sentence Length: 0, short; 1, average; 2, long;
    length = len(tokenizer.tokenize(sentence))
    if length < 12: # Length less than 12 chosen for 'short' sentences.
        return 0
    elif 12 <= length <= 25: # Length between 12 and 25 chosen for 'average' sentences
        return 1
    else: # Length > 25 chosen for 'long' sentences
        return 2

def get_para_loc_cat(sentence, paragraph_sentences):
    # Paragraph Location: 0, beginning; 1, middle; 2, end;
    for sentences in paragraph_sentences:
        if sentence not in sentences:
            continue
        for i, subsentence in enumerate(sentences):
            if sentence == subsentence:
                if i == 0:
                    return 0
                elif i == len(sentences) - 1:
                    return 2
                else:
                    return 1

def get_title_similarity(sentence, title):
    # Similarity to Title: 0, non-similar; 1, similar;
    token_count = 0
    for title_tokens in tokenizer.tokenize(title):
        for tokens in tokenizer.tokenize(sentence):
            if title_tokens == tokens:
                token_count += 1
                if token_count >= 2: # A sentence is similar to the title if it has two or more tokens in common.
                    return 1
    return 0

def get_contains_uppercase(sentence):
    # Uppercase Words: 0, no-uppercase; 1, uppercase;
    upper_count = 0
    for i, token in enumerate(tokenizer.tokenize(sentence)):
        if not token.islower() and i != 0:
            upper_count += 1
            if upper_count >= 2: # A sentence contains uppercases if there are >= 2 words (excluding the first word) that are uppercased.
                return 1
    return 0

def get_category(sentence, sentencized_summary):
    # Category of the Sentence: 0, non-summary; 1, summary;
    for summary_sentence in sentencized_summary:
        if sentence == summary_sentence:
            return 1
    return 0

if __name__ == "__main__":
    main()