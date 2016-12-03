"""This module does automated text classification ensuring AB tests for preprocessing techniques."""
import sys
import csv
import collections
import nltk as nt
import numpy as np
import statsmodels.stats.multicomp as multi 
import statsmodels.sandbox.stats.multicomp as multic
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from scipy import stats
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
# -*- coding: latin-1 -*-

def train_svm(x_data, y_data):
    """ Create and train the Support Vector Machine. """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    svm.fit(x_data, y_data)
    return svm

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
            stemmed_tokens = [nt.PorterStemmer().stem(t) for t in raw_tokens]
            final_tokens = ""
            for word in stemmed_tokens:
                if word not in stopwords.words('english'):
                    final_tokens += word + " "
            rows.append((final_tokens, classification))
        return rows, classes_counter

def generate_matrix(corpus):
    """This function prepares the attribute-value matrices."""
    data = [c[0] for c in corpus]
    labels = [c[1] for c in corpus]
    return data, labels

def vectorize_data(texts):
    """This function vectorizes text to matrices."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    return vectorizer.fit_transform(texts)


def classify_by_algorithm(classifier_name, x_test, y_test, kfold):
    """This function enables classification by a series of algorithms and train/test situation."""
    if classifier_name == 'SVM':
        classifier = train_svm(x_test, y_test)
    elif classifier_name == 'NBM':
        classifier = MultinomialNB().fit(x_test, y_test)
    elif classifier_name == 'C4.5':
        classifier = DecisionTreeClassifier().fit(x_test, y_test)
    prediction = classifier.predict(x_test)
    print(classifier_name)
    cross_result = cross_val_score(classifier, x_test, y_test, cv=kfold)
    accuracy = cross_result.mean()
    standard_deviation = cross_result.std()
    print("Acc: {0}".format(accuracy))
    print("Std: {0}".format(standard_deviation))
    return accuracy

def main():
    """Main method of the application."""
    print('Automated classfication of sentiment analysis datasets.')
    if len(sys.argv) != 4:
        print('Wrong parameter list. $1 = csv dataset; $2 = text fieldname; $3 category fieldname')
        quit()
    else:
        dataset, text_column, category_column = sys.argv[1], sys.argv[2], sys.argv[3]
        rows, classes_counter = treat_csv_dataset(dataset, text_column, category_column)
        majoritary_class = classes_counter.most_common()[0][1]/(sum(classes_counter.values()))
        rows_text, row_labels = generate_matrix(rows)
        x_raw = vectorize_data(rows_text)
        print("Majoritary class: {0}".format(majoritary_class))
        classifiers = ['SVM', 'NBM', 'C4.5']
        predictions = [] 
        x_train, x_test, y_train, y_test = train_test_split(x_raw, row_labels, test_size=0.1, random_state=0)
        kfold = StratifiedKFold(n_splits=10, shuffle = False)
        for classifier in classifiers:
            predictions.append(classify_by_algorithm(classifier, x_test, y_test, kfold))
        predictions = np.asarray(predictions)
        classifiers = np.asarray(classifiers)
        mc1 = multi.MultiComparison(predictions, classifiers)
        print(mc1.kruskal(multimethod='T'))
        m = multic.multipletests(mc1.data, alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
        print(m)
if __name__ == "__main__":
    main()
