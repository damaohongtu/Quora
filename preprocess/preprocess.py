#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: preprocess.py
@time: 18-12-24 下午11:29
@desc:
    1. Remove all irrelevant characters such as any non alphanumeric characters
    2. Tokenize your text by separating it into individual words
    3. Remove words that are not relevant, such as “@” twitter mentions or urls
    4. Convert all characters to lowercase, in order to treat words such as “hello”, “Hello”, and “HELLO” the same
    5. Consider combining misspelled or alternately spelled words to a single representation (e.g. “cool”/”kewl”/”cooool”)
    6. Consider lemmatization (reduce words such as “am”, “are”, and “is” to a common form such as “be”)
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import re

tqdm.pandas()

# 加载数据
train = pd.read_csv("../../Qura_data/train.csv")
test = pd.read_csv("../../Qura_data/test.csv")

# 大小写
train["question_text"] = train["question_text"].str.lower()
test["question_text"] = test["question_text"].str.lower()

train["question_text"] = train["question_text"].str.lower()
test["question_text"] = test["question_text"].str.lower()

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"\d+", "")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def removeNumbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])
    return text

# 删除连写
contraction_patterns = [(r'won\'t', 'will not'), (r'can\'t', 'can not'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),(r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

print("clean......")
train["question_text"] = train["question_text"].apply(lambda x: clean_text(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_text(x))

train["question_text"] = train["question_text"].apply(lambda x: removeNumbers(x))
test["question_text"] = test["question_text"].apply(lambda x: removeNumbers(x))

train = standardize_text(train,"question_text")
test = standardize_text(test,"question_text")

train.to_csv("../../Qura_data/train_clean.csv")
test.to_csv("../../Qura_data/test_clean.csv")





# ==========================================================================
print("embedding......")


# =========================================================
print("embedding glove")
file_glove = open('../../Qura_data/embeddings/glove.840B.300d/glove.840B.300d.txt')
embeddings_index = {}
for line in tqdm(file_glove):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file_glove.close()
with open("../../Qura_data/glove.pickle","wb") as f:
    pickle.dump(embeddings_index,f)
# ==========================================================
# print("embedding paragram")
# file_paragram = open('../../Qura_data/embeddings/paragram_300_sl999/paragram_300_sl999.txt')
# embeddings_index = {}
# for line in tqdm(file_paragram):
#     values = line.split(" ")
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# file_paragram.close()
# with open("../../Qura_data/paragram.pickle","wb") as f:
#     pickle.dump(embeddings_index,f)

# ===========================================================
print("embedding wiki")
file_wiki = open('../../Qura_data/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
next(file_wiki)
embeddings_index = {}
for line in tqdm(file_wiki):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
file_wiki.close()
with open("../../Qura_data/wiki.pickle","wb") as f:
    pickle.dump(embeddings_index,f)


