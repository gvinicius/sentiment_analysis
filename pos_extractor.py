import nltk

from nltk.corpus import wordnet
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

import os

import codecs

from nltk.corpus import treebank

path = 'gesteira_corpus/'


def generate_bases():
    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_path = subdir + os.path.sep + file
            text = codecs.open(file_path, 'r',encoding='utf-8', errors='ignore' )
            lowers = text.read().lower()
            tokenizer = RegexpTokenizer(r'\w+') 
            tokens = tokenizer.tokenize(lowers)
            filtered_tokens = [w for w in tokens if not w in stopwords.words('english')]
            tagged = nltk.pos_tag(tokens)
            selected_tokens = [word for word,pos in tagged if pos =='JJ' or pos =='RB' or pos =='CC' ]
            open("output/tagged_file".replace('file',file), "w").write(str(selected_tokens))

def main():
    generate_bases()
    print ("Selected only words of desired POS.")

if __name__ == "__main__":
    main()
