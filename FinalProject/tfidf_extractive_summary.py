import glob
import regex as re
from segment_sentences import segment_sentences
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    # The Corpus (List of all documents) and the text to be summarized (single document from corpus currently)
    corpus = [open(x, encoding='windows-1252').read() for x in glob.glob('data/raw_articles/*/*.txt')]  # TODO: Create a train/test split
    text = corpus[1]

    # Create the Tfidf vectorizer and fit the corpus to the vectorizer.
    vect = TfidfVectorizer(token_pattern=r"\$?\d*\.?\d+|\b\w\w+\b|\b[IiAa]\b")
    vect = vect.fit(corpus)

    # print(get_score(vect, text))

    # Print out the N (in this case 6) highest scores as the final summariation.
    # Scoring method is currently tf_idf score divided by the number of words in the sentence. 
    # TODO: Modify scoring method. Possibly add similarity to title? Other heruistics?
    print(text[0:text.find('\n')] + '\n-----') # Automatically prints title at top. Would prefer to not have.
    for i in range(6):
        print(get_score(vect, text)[i][1])


def get_score(vectorizer, document):
    scores = []

    title, sentences = segment_sentences(document, True)

    for sentence in sentences:
        score = 0
        word_count = 0
        for word in re.findall(r"\$?\d*\.?\d+|\b\w\w+\b|\b[IiAa]\b", sentence):
            score += vectorizer.vocabulary_[word.lower()]
            word_count += 1
        scores.append((score / word_count, sentence))

    scores.sort(reverse=True)
    return scores


if __name__ == '__main__':
    main()
