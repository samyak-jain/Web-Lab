import requests
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem.porter import *
import string
from nltk.corpus import wordnet as wn

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
    soup = BeautifulSoup(text, 'html.parser')
    for i in soup(['a', 'script', 'style']):
        i.extract()

    text = soup.get_text()
    translate_table = dict((ord(char), None) for char in string.punctuation)
    text = text.translate(translate_table)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english') + ['\n'])
    filtered = [w for w in tokens if w not in stop_words]
    wordnet_lemmatizer = WordNetLemmatizer()
    pos = dict(nltk.pos_tag(tokens))
    lem_words = [wordnet_lemmatizer.lemmatize(w, pos=penn_to_wn(pos.get(w))) if penn_to_wn(pos.get(w)) is not None else wordnet_lemmatizer.lemmatize(w) for w in filtered]
    words = set(lem_words)
    return words


if 0:
    print("Enter 4 URLs")
    sites = [input(str(i) + ": ") for i in range(4)]
else:
    # sites = [
    #  https://en.wikipedia.org/wiki/Web_mining ,  # 'http://www.vit.ac.in/',
    #     'https://www.dreamhost.com',
    #     'https://en.wikipedia.org/wiki/U.S._Snowboarding_Grand_Prix',
    #     'https://en.wikipedia.org/wiki/2002_FIU_Golden_Panthers_football_team'
    # ]
    sites = [
        'https://en.wikipedia.org/wiki/Web_mining',
        'https://www.quora.com/What-is-web-mining',
        'https://www.techopedia.com/definition/15634/web-mining',
        'https://www.educba.com/data-mining-vs-web-mining/'
    ]

htmls = [requests.get(i).text for i in sites]


indices = []
for count,i in enumerate(htmls):
    soup = BeautifulSoup(i, 'html.parser')
    for i in soup(['a', 'script', 'style']):
        i.extract()

    text = soup.get_text()
    translate_table = dict((ord(char), None) for char in string.punctuation)
    text = text.translate(translate_table)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english') + ['\n'])
    filtered = [w for w in tokens if w not in stop_words]
    wordnet_lemmatizer = WordNetLemmatizer()
    pos = dict(nltk.pos_tag(tokens))
    lem_words = [wordnet_lemmatizer.lemmatize(w, pos=penn_to_wn(pos.get(w))) if penn_to_wn(pos.get(w)) is not None else wordnet_lemmatizer.lemmatize(w) for w in filtered]
    for i in range(4):
        with open(str(i) + '.txt', 'w', encoding='utf-8') as f:
            f.write(text)
    words = set(lem_words)
    index = {}
    for i in words:
        index[i.encode('utf-8')] = [c for c,j in enumerate(lem_words) if j==i]

    print(index)
    indices.append(index)

    with open(str(count) + '_index.txt', 'w') as g:
        g.write(str(index))


terms = set()
term_index = {}
for i in htmls:
    terms = terms | preprocess(i)

for i in terms:
    term_index[i] = [{c: indices[c].get(i.encode())} for c,j in enumerate(htmls) if indices[c].get(i.encode()) is not None]

print(term_index)

with open("term_index.txt", "w") as g:
    g.write(str(term_index))
