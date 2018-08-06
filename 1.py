import nltk
from collections import Counter
import sys
from bs4 import BeautifulSoup as bs
import requests

def ext(x):
    soup = bs(x, "lxml")
    return soup.get_text()

if len(sys.argv) > 1:
    a = requests.get(sys.argv[1]).text
    x = ext(a)
else:
    with open("textfile.txt") as f:
        x = f.read()

print(f"The number of words is {len(x.split())}")
print(f"The number of sentences is " + str(x.count('\n')))
tokens = nltk.word_tokenize(x)
print("The count of the tokens is ")
print(Counter(tokens))
