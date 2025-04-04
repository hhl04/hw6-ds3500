

import json
from collections import Counter

def json_parser(filename):
    f = open(filename)
    raw = json.load(f)
    text = raw['text']
    words = text.split(" ")
    wc = Counter(words)  # Word count
    num = len(words)     # Actual number of words

    return {'wordcount': wc, 'numwords': num}



