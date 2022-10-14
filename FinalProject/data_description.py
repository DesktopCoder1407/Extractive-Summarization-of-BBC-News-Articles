import glob
import spacy

# Only includes lemmatizer, POS tagging, and detailed tag.
nlp = spacy.load('en_core_web_sm', exclude=['tok2vec', 'parser', 'ner'])
nlp.add_pipe('sentencizer', last=True)
nlp.max_length = 5100000


def directory_file_count():
    file_list = glob.glob('data\\raw_articles\\*\\*.txt')
    return f'File Count: {len(file_list)}'


def get_corpus():
    file_list = glob.glob('data/raw_articles/*/*.txt')
    corpus = ''''''
    for file in file_list:
        with open(file, 'rt', encoding='windows-1252') as file_in:
            corpus += file_in.read()
    return corpus


def get_corpus_split():
    file_list = glob.glob('data/raw_articles/*/')
    corpus = []
    for sub_folder in file_list:
        document = ''''''
        for file in glob.glob(sub_folder + '*.txt'):
            with open(file, 'rt', encoding='windows-1252') as file_in:
                document += file_in.read()
        corpus.append(document)
    return corpus


def sentences_count(corpus):
    doc = nlp(corpus)

    assert doc.has_annotation("SENT_START")
    return len(list(doc.sents))


full_corpus = get_corpus()
corpus_split = get_corpus_split()
print(f'Total sentences in the business category: {sentences_count(corpus_split[0])}')
print(f'Total sentences in the entertainment category: {sentences_count(corpus_split[1])}')
print(f'Total sentences in the politics category: {sentences_count(corpus_split[2])}')
print(f'Total sentences in the sport category: {sentences_count(corpus_split[3])}')
print(f'Total sentences in the tech category: {sentences_count(corpus_split[4])}')
print(f'Total sentences in the entire corpus: {sentences_count(full_corpus)}\n')


def tokens_count(corpus):
    doc = nlp(corpus)

    return len(list(doc))


print(f'Total tokens in the business category: {tokens_count(corpus_split[0])}')
print(f'Total tokens in the entertainment category: {tokens_count(corpus_split[1])}')
print(f'Total tokens in the politics category: {tokens_count(corpus_split[2])}')
print(f'Total tokens in the sport category: {tokens_count(corpus_split[3])}')
print(f'Total tokens in the tech category: {tokens_count(corpus_split[4])}')
print(f'Total tokens in the entire corpus: {tokens_count(full_corpus)}\n')


def vocabulary_count(corpus):
    doc = nlp(corpus)

    return len(set([token.lemma_ for token in doc]))


print(f'Vocabulary for the business category: {vocabulary_count(corpus_split[0])}')
print(f'Vocabulary for the entertainment category: {vocabulary_count(corpus_split[1])}')
print(f'Vocabulary for the politics category: {vocabulary_count(corpus_split[2])}')
print(f'Vocabulary for the sport category: {vocabulary_count(corpus_split[3])}')
print(f'Vocabulary for the tech category: {vocabulary_count(corpus_split[4])}')
print(f'Vocabulary for the entire corpus: {vocabulary_count(full_corpus)}')
