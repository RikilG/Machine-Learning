#!/bin/env python

import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as stop_words

class TextPreprocess:

    def __init__(self, data):
        assert type(data) == str
        self.raw_data = data
        self.data = data.lower()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set()
        self.dostemming = False
        self.dolemmatization = False

    def removeStopwords(self):
        self.stopwords.update(stop_words.words('english'))
        return self

    def removePunctuation(self):
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.punct_list = re.sub(r'(.)', r'\1 ', self.punctuation).split()
        self.stopwords.update(self.punct_list)
        self.data = re.sub(r'(['+self.punctuation+'])', r' \1 ', self.data)
        return self

    def stem(self):
        self.dostemming = True
        return self
    
    def lemmatize(self):
        self.dolemmatization = True
        return self

    def getNGrams(self, n=3):
        return self

    def getTokens(self):
        if len(self.stopwords) == 0:
            return self.data.split()
        self.words = word_tokenize(self.data)
        self.words = [word for word in self.words if word not in self.stopwords]
        if self.dolemmatization:
            self.words = [self.lemmatizer.lemmatize(word) for word in self.words]
        if self.dostemming:
            self.words = [self.stemmer.stem(word) for word in self.words]
        return self.words
