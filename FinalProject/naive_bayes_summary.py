from glob import glob
import re

# P(s in S | Features) = P(features | s in S) * P(sentence in Summary) [N/total_sentences]

def main():
    corpus_path = glob('data/raw_articles/business/*.txt')
    corpus_path.sort()
    corpus_y_path = glob('data/summarized_articles/business/*.txt')
    corpus_y_path.sort()

    corpus = [open(x, encoding='windows-1252').read() for x in corpus_path]
    corpus_y = [open(x, encoding='windows-1252').read() for x in corpus_y_path]

    title, doc = corpus[0].split('\n', maxsplit=1)
    doc_y = corpus_y[0]

    for vector in doc_to_feature_vector(title, doc, doc_y):
        print(vector[0])


def doc_to_feature_vector(title, doc, doc_y):
    doc_features = [] # FORMAT: SentenceLength, ParagraphLocation, SimilarityToTitle, UppercaseWords, CATEGORY
    
    paragraphs = re.findall(r".+", doc)
    sentences = re.findall(r"[^ \n].+?\.(?!\d)", doc)

    for sentence in sentences:
        features = []
        
        # Sentence Length
        sentence_len = len(re.findall(r"\S?\d+[.,]\d+\w+|[^ \n,.]+", sentence))
        if sentence_len < 12:
            features.append('short')
        elif 12 <= sentence_len <= 25:
            features.append('average')
        else:
            features.append('long')
        
        # Paragraph Location
        for paragraph in paragraphs:
            p_sents = re.findall(r"[^ \n].+?\.(?!\d)", paragraph)
            for i, subsentence in enumerate(p_sents):
                if sentence == subsentence:
                    if i == 0:
                        features.append('beginning')
                    elif i == len(p_sents) - 1:
                        features.append('end')
                    else:
                        features.append('middle')
        
        doc_features.append((features, sentence))
    return doc_features


if __name__ == "__main__":
    main()

# Similarity to Title: Contains >= 2 words in title, contains no words in title
# Uppercase Words: Sentence contains >= 1(2) words that have an uppercase somewhere (ignore first word), contains no uppercase
# ?Fixed Phrase: Sentence contains a phrase that is prespecified.