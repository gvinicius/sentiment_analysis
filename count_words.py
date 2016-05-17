import nltk
import sys
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import csv
from nltk.tokenize import RegexpTokenizer

import time

import os

import codecs

from nltk.corpus import treebank
# -*- coding: latin-1 -*-

#path = 'gesteira_corpus/'
"""
path = '../dataset/Base-Canada-{0}'.format(sys.argv[1])
destination_path = "../dataset/output-g-{0}".format(sys.argv[1])
"""
destination_path_file = '../dataset/economics/result/file'



def generate_bases():
    selected_tokens = []
    with open('../dataset/economics/weather-evaluated-agg-DFE.csv', 'r', encoding="latin1") as csvfile:
        csv_reader = csv.reader(csvfile, quotechar='\"')
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row['tweet_text']
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(text)
            tagged = nltk.tag.pos_tag(raw_tokens)
            selected_tokens_1 = [word for word,pos in tagged if pos in ['JJ', 'NN', 'NNP', 'RBR' ] ]
            selected_tokens.append(selected_tokens_1)
        print(len(selected_tokens))

def main():
    generate_bases()
    print ("Selected only words of desired POS.")
if __name__ == "__main__":
    main()
