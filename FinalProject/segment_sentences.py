from nltk import sent_tokenize
import regex as re


def segment_sentences(document: str, find_title: bool = False):
    title = None
    paragraphs = re.split(r'\n+', document)
    sentences = []

    if find_title:
        sentences = sent_tokenize(paragraphs[0])
        if len(sentences) == 1:
            title = sentences[0]
            paragraphs = paragraphs[1:]

    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))

    if find_title:
        return title, sentences
    return sentences
