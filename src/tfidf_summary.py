import tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def score_sentences(corpus: list[str], text: str, n: int, vect:TfidfVectorizer|None = None):
    scored_sentences = []

    # Create the Tfidf vectorizer and fit the corpus to the vectorizer.
    if vect is None:
        vect = TfidfVectorizer(token_pattern=r"\S?\d+[.,]\d+\w+|[^ \n,.]+")
        vect = vect.fit(corpus)
    
    # Iterates through each sentence, getting the tf_idf score for each word and finding the average tf_idf score
    # to use as the score value for the sentence.
    for i, sentence in enumerate(tokenizer.sentencize(text)):
        score = 0
        word_count = 1
        for word in tokenizer.tokenize(sentence):
            score += vect.vocabulary_[word.lower()]
            word_count += 1
        scored_sentences.append((score / word_count, (i, sentence)))

    # Sort the sentences by their scores (Highest to lowest) and select the top n.
    scored_sentences.sort(reverse=True)
    scored_sentences = [x[1] for x in scored_sentences[:n]]

    # Sort the chosen sentences by the order in which they appear in the original document.
    scored_sentences.sort()
    scored_sentences = [x[1] for x in scored_sentences]
    return scored_sentences

def score_corpus(training_corpus: list[str], testing_corpus: list[str], n: int):
    summaries = []

    # Create the Tfidf vectorizer and fit the corpus to the vectorizer.
    vect = TfidfVectorizer(token_pattern=r"\S?\d+[.,]\d+\w+|[^ \n,.]+")
    vect = vect.fit(training_corpus)

    # Summarize each document within the testing corpus.
    for doc in testing_corpus:
        doc_summary = ''
        for sentence in score_sentences(testing_corpus, doc, n, vect):
            doc_summary += sentence + ' '
        summaries.append(doc_summary[:-1])

    return summaries