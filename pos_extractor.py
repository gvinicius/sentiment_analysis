import nltk
import sys
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

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
            classification = "noncfb" if "noncfb" in os.path.dirname(subdir) else "cfb"
            file_path = subdir + os.path.sep + file
            text = codecs.open(file_path, 'r',encoding='ascii', errors='ignore' )
            lowers = text.read().lower()
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(lowers)
            tagged = nltk.tag.pos_tag(raw_tokens)
            selected_tokens = [word for word,pos in tagged if pos =='JJ' or (pos =='CC' and word in ['but', 'yet', 'still', 'although', 'however']) ]
            final_tokens = " "
            for word in selected_tokens:
                final_tokens += word +" "
            filename = str(counter) + "_" + classification
            counter += 1
            print '{0}\r'.format(counter/len(files)),
            print
            open(destination_path_file.replace('file',filename), "w").write(str(final_tokens))

def main():
    generate_bases()
    print ("Selected only words of desired POS.")

if __name__ == "__main__":
    delection_command = 'rm -Rf ' destination_path + '/*'

    main()
