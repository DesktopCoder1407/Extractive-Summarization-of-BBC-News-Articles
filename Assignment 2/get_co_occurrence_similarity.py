import pandas
import numpy
import math


def cosine_similarity(word1: str, word2: str):
    numer = numpy.dot(data[word1], data[word2])
    denom = math.sqrt(numpy.dot(data[word1], data[word1])) * math.sqrt(numpy.dot(data[word2], data[word2]))

    return round(numer/denom, 4)


data = pandas.read_csv('co_occurrence.csv', index_col=['Count'])

output_data = dict()
for word1 in data:
    output_data[word1] = dict()
    for word2 in data:
        output_data[word1][word2] = cosine_similarity(word1, word2)

output_df = pandas.DataFrame.from_dict(output_data)
output_df.to_csv('co_occurrence_similarity.csv')
