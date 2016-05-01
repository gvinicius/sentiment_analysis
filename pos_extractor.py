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
    counter = 1
    with open('../dataset/economics/Full-Economic-News-DFE-839861.csv', 'rU') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row['headline'] + ' ' + row['text']
            classification = row['relevance']
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(text)
            final_tokens = ""
            for word in raw_tokens:
                final_tokens += word +" "
            filename = str(counter) + "_" + classification
            counter += 1
            print (counter)
            open(destination_path_file.replace('file',filename), "w").write(str(final_tokens))


def main():
    generate_bases()
    print ("Selected only words of desired POS.")
    # run_classification = 'bash preprocesssing.sh ' + sys.argv[1]
    # os.system(run_classification)
if __name__ == "__main__":
    main()
