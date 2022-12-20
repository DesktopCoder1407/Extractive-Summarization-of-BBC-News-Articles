# TOKENIZER REGEX: r"\S?\d+[.,]\d+\w+|[^ \n,.]+"
# PARAGRAPH REGEX: r".+"
# SENTENCE REGEX: r"[^ \n].+?\.(?!\d)"

import re

def tokenize(doc: str):
    return re.findall(r"\S?\d+[.,]\d+\w+|[^ \n,.]+", doc)

def sentencize(doc: str):
    return re.findall(r"[^ \n].+?\.(?!\d)", doc)

def paragraphize(doc: str):
    return re.findall(r".+", doc)