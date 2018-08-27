from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet as wn
import ast
import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

def preprocess(text):
    translate_table = dict((ord(char), None) for char in string.punctuation)
    text = text.translate(translate_table)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english') + ['\n'])
    filtered = [w for w in tokens if w not in stop_words]
    wordnet_lemmatizer = WordNetLemmatizer()
    pos = dict(nltk.pos_tag(tokens))
    lem_words = [wordnet_lemmatizer.lemmatize(w, pos=penn_to_wn(pos.get(w))) if penn_to_wn(pos.get(w)) is not None else wordnet_lemmatizer.lemmatize(w) for w in filtered]
    words = set(lem_words)
    index = {}
    for i in words:
        index[i.encode('utf-8')] = [c for c,j in enumerate(lem_words) if j==i]

    return index

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([len(vec1[x]) * len(vec2[x]) for x in intersection])

     sum1 = sum([len(vec1[x])**2 for x in vec1.keys()])
     sum2 = sum([len(vec2[x])**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     print(vec2.keys())

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def euc(vec1, vec2):
    union = set(vec1.keys()) | set(vec2.keys())
    numerator = sum([(len(vec1.get(x, [])) - len(vec2.get(x, [])))**2 for x in union])**(1/2)
    denominator = len(vec1.keys()) * len(vec2.keys())

    if not denominator:
        return 0.0
    else:
        return numerator / denominator

vectorizer = TfidfVectorizer()
indices = []
for i in range(4):
    with open(str(i) + '_index.txt') as f:
        indices.append(f.read())

query = input("Enter Query: ")
cosines = [get_cosine(preprocess(query), ast.literal_eval(i)) for i in indices]
print(cosines)

print("The ranking based on cosine similarity is")
for i in np.asarray(cosines).argsort()[:-len(cosines)-1:-1]:
    print(i+1)

eucs = [euc(preprocess(query), ast.literal_eval(i)) for i in indices]
print(eucs)

print("The ranking based on Euclidean Distance is")
for i in np.asarray(eucs).argsort()[:-len(eucs)-1:-1][::-1]:
    print(i+1)
