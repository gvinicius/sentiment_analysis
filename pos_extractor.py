import nltk

from nltk.corpus import wordnet
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

import os

import codecs

from nltk.corpus import treebank


#path = 'gesteira_corpus/'
path = '../dataset/test/'

def generate_bases():
    for subdir, dirs, files in os.walk(path):
        counter = 1
        for file in files:
            file_path = subdir + os.path.sep + file
            text = codecs.open(file_path, 'r',encoding='utf-8', errors='ignore' )
            lowers = text.read().lower()
            tokens = " "
            classification = ""
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(lowers)
            for word in raw_tokens:
                if (tokens == " " and classification == ""): # or word == "cfb":
                    classification += word
                else:
                    tokens += str(word) + " "
            new_tokens = tokenizer.tokenize(tokens)
#            classification = tokens.partition(' ')[0]
#            tokens.replace("cfb", "", 1)
#            filtered_tokens = [w for w in tokens if not w in stopwords.words('english')]
            tagged = nltk.tag.pos_tag(new_tokens)
            selected_tokens = [word for word,pos in tagged if pos =='JJ' or pos =='RB' or pos =='CC' and word != "and" ]
            final_tokens = " "
            for word in selected_tokens:
                final_tokens += word +" "
            filename = str(counter) + "_" + classification
            counter += 1
            open("output/file".replace('file',filename), "w").write(str(final_tokens))


def main():
    generate_bases()
    print ("Selected only words of desired POS.")

if __name__ == "__main__":
    main()
