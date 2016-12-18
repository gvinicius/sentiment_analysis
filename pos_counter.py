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
            pos = [str(pos) for word, pos in tagged]
            rows.append((' '.join(pos), classification))
        return rows

def vectorize_data(texts):
    """This function vectorizes text to matrices."""
    vectorizer = TfidfVectorizer()
    transformation = vectorizer.fit_transform(texts)
    indices = np.argsort(vectorizer.idf_)[::1]
    features = vectorizer.get_feature_names()
    top_n = 11
    top_features = [features[i] for i in indices[:top_n]]
    return top_features, features

def generate_matrix(corpus):
    """This function prepares the attribute-value matrices."""
    data = [c[0] for c in corpus]
    labels = [c[1] for c in corpus]
    return data, labels

def main():
    """Main method of the application."""
    print('Automated classfication of sentiment analysis datasets.')
    if len(sys.argv) != 4:
        print('Wrong parameter list. $1 = csv dataset; $2 = text fieldname')
        quit()
    else:
        rows = treat_csv_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
        rows_text, labels = generate_matrix(rows)
        top, features = vectorize_data(rows_text)
        print(features)
        print(top)
if __name__ == "__main__":
    main()
