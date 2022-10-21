import glob
import nltk
import numpy
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    # Document (Input for testing, basically the thing being summarized)
    text = '''Dollar gains on Greenspan speech

The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise.

And Alan Greenspan highlighted the US government's willingness to curb spending and rising household savings as factors which may help to reduce it. In late trading in New York, the dollar reached $1.2871 against the euro, from $1.2974 on Thursday. Market concerns about the deficit has hit the greenback in recent months. On Friday, Federal Reserve chairman Mr Greenspan's speech in London ahead of the meeting of G7 finance ministers sent the dollar higher after it had earlier tumbled on the back of worse-than-expected US jobs data. "I think the chairman's taking a much more sanguine view on the current account deficit than he's taken for some time," said Robert Sinche, head of currency strategy at Bank of America in New York. "He's taking a longer-term view, laying out a set of conditions under which the current account deficit can improve this year and next."

Worries about the deficit concerns about China do, however, remain. China's currency remains pegged to the dollar and the US currency's sharp falls in recent months have therefore made Chinese export prices highly competitive. But calls for a shift in Beijing's policy have fallen on deaf ears, despite recent comments in a major Chinese newspaper that the "time is ripe" for a loosening of the peg. The G7 meeting is thought unlikely to produce any meaningful movement in Chinese policy. In the meantime, the US Federal Reserve's decision on 2 February to boost interest rates by a quarter of a point - the sixth such move in as many months - has opened up a differential with European rates. The half-point window, some believe, could be enough to keep US assets looking more attractive, and could help prop up the dollar. The recent falls have partly been the result of big budget deficits, as well as the US's yawning current account gap, both of which need to be funded by the buying of US bonds and assets by foreign firms and governments. The White House will announce its budget on Monday, and many commentators believe the deficit will remain at close to half a trillion dollars.'''

    corpus = [open(x).read() for x in glob.glob('data/raw_articles/*/*.txt')]  # The training corpus TODO: Set a % of stuff being trained

    # Create the Tfidf vectorizer and fit the corpus to the vectorizer.
    # TODO: Possibly use different (more accurate) regex expressions, possibly r'''(?u)\b\w+\b|\$\d*\.?\d+'''
    vect = TfidfVectorizer(token_pattern=r"\$?\d*\.?\d+|\b\w\w+\b|\b[IiAa]\b")
    vect = vect.fit(corpus)

    # print(get_score(vect, text))


    print(text[0:text.find('\n')] + '\n-----')
    for i in range(5):
        print(get_score(vect, text)[i][1])
    # print(vect.vocabulary_)

    # X = vect.transform(nltk.sent_tokenize(text))  # Summarize the document
    #
    # # Debug printing
    # print(vect.get_feature_names_out())
    # print(top_n_vector_locations(X.toarray(), 3))
    #
    # # Find the n top vectors (scored via summing) and print them out. TODO: Find different scoring method
    # for loc in top_n_vector_locations(X.toarray(), 5):
    #     print(nltk.sent_tokenize(text)[loc])


def get_score(vectorizer, document):
    title = document[0:document.find('\n')]
    # title_score = 0

    paragraphs = re.split(r'\n\n|\n', document)
    sentences = []
    scores = []

    for paragraph in paragraphs:
        sentences.extend(nltk.sent_tokenize(paragraph))

    for sentence in sentences:
        score = 0
        word_count = 0
        for word in re.findall(r"\$?\d*\.?\d+|\b\w\w+\b|\b[IiAa]\b", sentence):
            score += vectorizer.vocabulary_[word.lower()]
            word_count += 1
        scores.append((score / word_count, sentence))

    scores.sort(reverse=True)
    return scores


def top_n_vector_locations(sentence_vectors, n, print_score = False):
    top = []

    # Find n total vectors
    for i in range(n):
        max_vector = -1
        max_value = 0

        # Loop through each vector and find the one with the largest sum
        for vector_i in range(len(sentence_vectors)):
            if sum(sentence_vectors[vector_i]) > max_value:
                max_value = sum(sentence_vectors[vector_i])
                max_vector = vector_i

        # Add the best vector and remove it from further consideration.
        top.append(max_vector + i)
        sentence_vectors = numpy.delete(sentence_vectors, max_vector, axis=0)
    return top


if __name__ == '__main__':
    main()
