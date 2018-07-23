import nltk
from collections import Counter

with open("textfile.txt") as f:
    x = f.read()
    tokens = nltk.word_tokenize(x)
    print(Counter(tokens))
