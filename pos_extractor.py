import nltk
import sys
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

import time

import os

import codecs

from nltk.corpus import treebank
# -*- coding: latin-1 -*-

#path = 'gesteira_corpus/'
path = '../dataset/Base-Canada-{0}'.format(sys.argv[1])
destination_path = "../dataset/output-g-{0}".format(sys.argv[1])
destination_path_file = destination_path + '/file'

def generate_bases():
    counter = 1
    for subdir, dirs, files in os.walk(path):
        for file in files:
            classification = "noncfb" if "non-CFB" in os.path.dirname(subdir) else "cfb"
            file_path = subdir + os.path.sep + file
            text = codecs.open(file_path, 'r',encoding='utf-8', errors='ignore' )
            lowers = text.read().lower()
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(lowers)
            tagged = nltk.tag.pos_tag(raw_tokens)
            selected_tokens = [word for word,pos in tagged if pos in ['JJ','RB', 'JJR ', 'JJS', 'JJT', 'RBR', 'RN', 'RP' ] or (pos =='CC' and word in ['but', 'yet', 'still', 'although', 'however']) ]
            final_tokens = ""
            for word in selected_tokens:
                final_tokens += word +" "
            filename = str(counter) + "_" + classification
            counter += 1
            print (counter)
            open(destination_path_file.replace('file',filename), "w").write(str(final_tokens))

def generate_bases_common():
    counter = 1
    for subdir, dirs, files in os.walk(path):
        for file in files:
            classification = "noncfb" if "non-CFB" in os.path.dirname(subdir) else "cfb"
            file_path = subdir + os.path.sep + file
            text = codecs.open(file_path, 'r',encoding='utf-8', errors='ignore' )
            lowers = text.read().lower()
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(lowers)
            final_tokens = ""
            for word in raw_tokens:
                final_tokens += word +" "
            filename = str(counter) + "_" + classification
            counter += 1
            print (counter)
            open(destination_path_file.replace('file',filename), "w").write(str(final_tokens))

def main():
    delection_command = 'rm -Rf ' + destination_path + '/* -v'
    os.system(delection_command)
    if(sys.argv[2]=='tag'):
        generate_bases()
    elif(sys.argv[2]=='nontag'):
        generate_bases_common()
    print ("Selected only words of desired POS.")
    run_classification = 'bash preprocesssing.sh ' + sys.argv[1]
    os.system(run_classification)
if __name__ == "__main__":
    main()
