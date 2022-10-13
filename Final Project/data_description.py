import os
import glob
#import spacey
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')


def directory_file_count():
    file_list = glob.glob('data\\raw_articles\\*\\*.txt')
    return f'File Count: {len(file_list)}'


def get_corpus():
    file_list = glob.glob('data\\raw_articles\\*\\*.txt')
    corpus = ''''''
    for file in file_list:
        with open(file, 'rt') as file_in:
            corpus += file_in.read()
    return corpus


def get_corpus_split():
    file_list = glob.glob('data\\raw_articles\\*\\')
    corpus = []
    for sub_folder in file_list:
        for file in glob.glob(sub_folder + '*.txt'):
            with open(file, 'rt') as file_in:
                corpus.append(file_in.read())
    return corpus

#WORK ON FROM HERE DOWN
def directory_sentences_count():
    print('Sentences Count:')
    total_count = 0
    for sub_directory in os.listdir(root_directory):
        print(sub_directory)
        sentence_count = 0
        files = [root_directory + '/' + sub_directory + '/' + x for x in os.listdir(root_directory + '/' + sub_directory + '/')]
        for filename in files:
            file = open(filename, 'r')
            tokenizer = sent_tokenize(file.read())
            sentence_count += len(tokenizer)
        print(sentence_count)
        total_count += sentence_count
    print('total')
    print(total_count)


def directory_tokens_count():
    print('Token Count:')
    total_count = 0
    for sub_directory in os.listdir(root_directory):
        print(sub_directory)
        token_count = 0
        files = [root_directory + '/' + sub_directory + '/' + x for x in os.listdir(root_directory + '/' + sub_directory + '/')]
        for filename in files:
            file = open(filename, 'r')
            tokenizer = word_tokenize(file.read())
            token_count += len(tokenizer)
        print(token_count)
        total_count += token_count
    print('total')
    print(total_count)


def directory_vocabulary_count():
    print('Vocabulary Count:')
    total_count = 0
    for sub_directory in os.listdir(root_directory):
        print(sub_directory)
        type_count = 0
        files = [root_directory + '/' + sub_directory + '/' + x for x in os.listdir(root_directory + '/' + sub_directory + '/')]
        for filename in files:
            file = open(filename, 'r')
            tokenizer = word_tokenize(file.read())
            type_count += len(set(tokenizer))
        print(type_count)
        total_count += type_count
    print('total')
    print(total_count)

