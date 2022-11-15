import re
from sklearn.feature_extraction.text import TfidfVectorizer

def score_sentences(corpus, doc, n):
    scored_sentences = []

    # Create the Tfidf vectorizer and fit the corpus to the vectorizer.
    vect = TfidfVectorizer(token_pattern=r"\$?\d*\.?\d+|\b\w\w+\b|\b[IiAa]\b")
    vect = vect.fit(corpus)
    
    # Iterates through each sentence, getting the tf_idf score for each word and finding the average tf_idf score
    # to use as the score value for the sentence.
    for i, sentence in enumerate(doc.sents):
        score = 0
        word_count = 0
        for word in re.findall(r"\$?\d*\.?\d+|\b\w\w+\b|\b[IiAa]\b", str(sentence)):
            score += vect.vocabulary_[word.lower()]
            word_count += 1
        scored_sentences.append((score / word_count, (i, str(sentence)))) if word_count > 0 else scored_sentences

    # Sort the sentences by their scores (Highest to lowest) and select the top n.
    scored_sentences.sort(reverse=True)
    scored_sentences = [x[1] for x in scored_sentences[:n]]

    # Sort the chosen sentences by the order in which they appear in the original document.
    scored_sentences.sort()
    scored_sentences = [x[1] for x in scored_sentences]
    return scored_sentences
