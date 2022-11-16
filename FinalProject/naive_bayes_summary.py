from glob import glob
import numpy
import tokenizer
from sklearn.naive_bayes import CategoricalNB

def main():
    corpus_path = glob('data/raw_articles/business/*.txt')
    corpus_path.sort()
    corpus_y_path = glob('data/summarized_articles/business/*.txt')
    corpus_y_path.sort()

    corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]
    corpus_y = [open(x, encoding='windows-1252').read() for x in corpus_y_path]

    print(get_training_vector(corpus, corpus_y))
    exit()

    # doc = corpus[0]
    # doc_y = corpus_y[0]

    # vector = doc_to_feature_vector(doc, doc_y)

    # print(numpy.array(vector)[3,0])
    # exit()

    vector_only = numpy.array([x[0] for x in vector])

    classifier = CategoricalNB()
    classifier.fit(vector_only[:, :-1], vector_only[:, -1])
    
    for t, p in zip(numpy.array(vector)[0:20,2], classifier.predict(vector_only[0:20, :-1])):
        if p == 1:
            print(t)
    # print(numpy.array(vector)[0:20,2])
    # print(classifier.predict(vector_only[0:20, :-1]))

    # print(classifier.score(vector_only[:, :-1], vector_only[:, -1]))

def get_training_vector(corpus, gold_corpus):
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

# ?Fixed Phrase: Sentence contains a phrase that is prespecified.