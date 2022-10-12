import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

root_directory = 'data/raw_articles/'


def directory_file_count():
    print('File Count:')
    for sub_directory in os.listdir(root_directory):
        print(sub_directory)
        file_count = 0
        for filename in os.listdir(os.path.join(root_directory, sub_directory)):
            file_count += 1
        print(file_count)


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

