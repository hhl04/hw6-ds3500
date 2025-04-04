

"""
File: textastic.py

Description: A reusable, extensible framework
for comparitive text analysis designed to work
with any arbitrary collection of related documents.

"""


from collections import Counter, defaultdict
import random as rnd
import matplotlib.pyplot as plt



class Textastic:

    def __init__(self):
        """ Contructor """
        self.data = defaultdict(dict)

    def simple_text_parser(self, filename):
        """ For processing simple, unformatted text documents """

        # harcoded results for demo purposes
        results = {
            'wordcount': Counter("to be or not to be".split(" ")),
            'numwords': rnd.randrange(10, 50)
        }

        print("Parsed: ", filename, ": ", results)
        return results



    def load_text(self, filename, label=None, parser=None):
        """ Register a document with the framework and
        store data extracted from the document to be used
        later in visualizations """

        results = self.simple_text_parser(filename) # default
        if parser is not None:
            results = parser(filename)

        if label is None:
            label = filename

        # Store results in the data dictionary (self.data)
        # coming soon.....

        # results = {
        #     'wordcount': Counter("to be or not to be".split(" ")),
        #     'numwords': rnd.randrange(10, 50)
        # }

        for k, v in results.items():
            self.data[k][label] = v

        # {numwords: 10, ...} --->  numwords: {label:10, label2: 20, .....}


    def compare_num_words(self):
        """ A very simplistic visualization that creats a bar
        chart comparing the  number of words in each file.
        (Not useful for your poster!!!!) """

        num_words = self.data['numwords']
        for label, nw in num_words.items():
            plt.bar(label, nw)
        plt.show()

