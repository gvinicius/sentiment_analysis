"""This module calculates tags frequency into a textual dataset."""
import sys
import csv
import collections
import nltk as nt
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def treat_csv_dataset(dataset, text_field, category_field):
    """This function does preprocessing upon csv dataset files."""
    classes_counter = collections.Counter()
    with open(dataset, 'r', encoding="latin1") as csvfile:
        rows = []
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = BeautifulSoup(row[text_field], 'html5lib').get_text()
            classification = str(row[category_field])
            classes_counter[classification] += 1
            raw_tokens = RegexpTokenizer(r'\w+').tokenize(text)
            tagged = nt.tag.pos_tag(raw_tokens)
            final_pos = ""
            for word in raw_tokens:
                if word not in stopwords.words('english'):
                    final_tokens += nt.PorterStemmer().stem(word) + " "
            rows.append((final_tokens, classification))
        return rows, classes_counter

def vectorize_data(texts):
    """This function vectorizes text to matrices."""
    vectorizer = TfidfVectorizer()
    transformation = vectorizer.fit_transform(texts)
    dimensionality_notion = len(transformation.toarray()[0])
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    top_n = 2
    top_features = [features[i] for i in indices[:top_n]]
    return transformation, dimensionality_notioni, top_feature

def main():
    """Main method of the application."""
    print('Automated classfication of sentiment analysis datasets.')
    if len(sys.argv) != 3:
        print('Wrong parameter list. $1 = csv dataset; $2 = text fieldname')
        quit()
    else:
        treat_csv_dataset(dataset, text_field, category_field)
        print(vectorize_data(texts))
if __name__ == "__main__":
    main()
