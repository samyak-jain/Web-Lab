import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter
from nltk.stem.porter import *

with open('htmlfile.txt') as f:
    x = f.read()
    soup = BeautifulSoup(x, 'html.parser')
    text = soup.get_text()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    stemmer = nltk.stemmer.PorterStemmer()
    stemmed_words = list(map(lambda x: stemmer.stem(x), filtered))
    print(Counter(stemmed_words))
