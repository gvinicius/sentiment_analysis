import nltk
import sys
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

import time

import os

import codecs

from nltk.corpus import treebank
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
# -*- coding: latin-1 -*-

#path = 'gesteira_corpus/'
path = '../dataset/Base-Canada-{0}'.format(sys.argv[1])
destination_path = "../dataset/output-g-{0}".format(sys.argv[1])
destination_path_file = destination_path + '/file'

def generate_bases():
#    with open('../dataset/economics/nuclear.csv', 'r', encoding="latin1") as csvfile:
#    with open('../dataset/economics/nuclear.csv', 'r', encoding="latin1") as csvfile:


def main():
    print ("Selected only words of desired POS.")
    rows = []
    with open('../dataset/economics/progressive-tweet-sentiment.csv', 'r', encoding="latin1") as csvfile:
        csv_reader = csv.reader(csvfile, quotechar='\"')
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row['tweet_text']
            classification = str(row['sentiment']).replace(" ","_").replace("/","").replace("'","")
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(text)
            tagged = nltk.tag.pos_tag(raw_tokens)
            selected_tokens = [word for word,pos in tagged if pos in ['JJR','JJS','NNS','NNP'] ]
#            selected_tokens = [word for word,pos in tagged if pos in ['RBR','JJR','JJS','NNS','NNP'] ]
#            selected_tokens = raw_tokens
            final_tokens = ""
            for word in selected_tokens:
                final_tokens += word + " " 
            rows.append((final_tokens, classification))
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(rows[0])
#    run_classification = 'bash preprocesssing.sh ' + sys.argv[1]
#    os.system(run_classification)
if __name__ == "__main__":
    main()
