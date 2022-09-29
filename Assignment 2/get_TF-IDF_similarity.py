import pandas
import numpy
import math


def cosine_similarity(word1: str, word2: str):
    numer = numpy.dot(data.T[word1], data.T[word2])
    denom = math.sqrt(numpy.dot(data.T[word1], data.T[word1])) * math.sqrt(numpy.dot(data.T[word2], data.T[word2]))

    if denom == 0:
        return 0
    else:
        return round(numer/denom, 4)


data = pandas.read_csv('tf-idf.csv', index_col=['Word'])

output_data = dict()
for word1 in data.T:
    output_data[word1] = dict()
    for word2 in data.T:
        output_data[word1][word2] = cosine_similarity(word1, word2)

output_df = pandas.DataFrame.from_dict(output_data)
output_df.to_csv('tf-idf_similarity.csv')
