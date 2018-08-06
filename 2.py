import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter
from nltk.stem.porter import *
import string

with open('htmlfile.txt') as f:
    x = f.read()
    soup = BeautifulSoup(x, 'html.parser')
    soup.script.decompose()
    soup.a.decompose()
    soup.style.decompose()
    text = soup.get_text()
    print("The number of lines is " + str(text.count('\n')))
    translate_table = dict((ord(char), None) for char in string.punctuation)
    text = text.translate(translate_table)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    print(f"The number of words after stopword removal is {len(Counter(filtered))}")
    print("The frequency count after stopword removal is ")
    print(Counter(filtered))
    stemmer = PorterStemmer()
    stemmed_words = list(map(lambda x: stemmer.stem(x), filtered))
    print(f"\n\nThe number of words after stemming is {len(Counter(stemmed_words))}")
    print("The frequency count after stemming is ")
    print(Counter(stemmed_words))
