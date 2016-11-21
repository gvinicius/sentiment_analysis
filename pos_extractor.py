import nltk
import sys
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import treebank
from nltk.tokenize import RegexpTokenizer
import csv
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import tree
# -*- coding: latin-1 -*-
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

def treat_csv_dataset(dataset, preprocessing_param):
    with open(dataset, 'r', encoding="latin1") as csvfile:
        rows = []
        csv_reader = csv.reader(csvfile, quotechar='\"')
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row['tweet_text']
            classification = str(row['sentiment']).replace(" ","_").replace("/","").replace("'","")
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(text)
            tagged = nltk.tag.pos_tag(raw_tokens)
            if preprocessing_param == 'tags':
                selected_tokens = [word for word,pos in tagged if pos in ['JJR','JJS','NNS','NNP'] ]
            else:
                selected_tokens = raw_tokens
            final_tokens = ""
            for word in selected_tokens:
                final_tokens += word + " " 
            #final_tokens = tokenizer.tokenize(final_tokens)
            rows.append((final_tokens, classification))
        return rows

def generate_matrix(corpus):
    data = [c[0] for c in corpus]
    labels = [c[1] for c in corpus]
    return data, labels

def main():
    print ("Selected only words of desired POS.")
    rows = treat_csv_dataset(sys.argv[1], sys.argv[2])
    rows_text, rows_label = generate_matrix(rows)
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(rows_text)
    X_train, X_test, y_train, y_test = train_test_split(X, rows_label, test_size=0.2)
    svm = train_svm(X_train, y_train)
    pred = svm.predict(X_test)
    print('SVM')
    print(confusion_matrix(pred, y_test))
    print(svm.score(X_test, y_test))
    quit()
    print('NBM')
    clf = MultinomialNB().fit(X_test, y_test)
    pred = clf.predict(X_test)
    print(confusion_matrix(pred, y_test))
    print(clf.score(X_test, y_test))
    print('C4.5')
    c45 = tree.DecisionTreeClassifier()
    c45 = c45.fit(X_test, y_test)
    pred = c45.predict(X_test)
    print(confusion_matrix(pred, y_test))
    print(c45.score(X_test, y_test))
if __name__ == "__main__":
    main()
