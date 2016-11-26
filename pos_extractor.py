"""This module does automated text classification ensuring AB tests for preprocessing techniques."""
import sys
import csv
import collections
import nltk as nt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
# -*- coding: latin-1 -*-

def train_svm(x_data, y_data):
    """ Create and train the Support Vector Machine. """
    svm = SVC(C=1000000.0, kernel='rbf')
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

def classify_by_algorithm(classifier_name, x_situation, y_situation):
    """This function enables classification by a series of algorithms and train/test situation."""
    if classifier_name == 'SVM':
        classifier = train_svm(x_situation, y_situation)
    elif classifier_name == 'NBM':
        classifier = MultinomialNB().fit(x_situation, y_situation)
    elif classifier_name == 'C4.5':
        classifier = DecisionTreeClassifier().fit(x_situation, y_situation)
    prediction = classifier.predict(x_situation)
    print(classifier_name)
    print(confusion_matrix(prediction, y_situation))
    print(classification_report(prediction, y_situation))
    print("Score:{0}".format(classifier.score(x_situation, y_situation)))
    kfold = KFold(n_splits=10)
    print("Acc.:{0}".format(cross_val_score(classifier, x_situation, y_situation, cv=kfold).mean()))
    print("Std.:{0}".format(cross_val_score(classifier, x_situation, y_situation, cv=kfold).std()))
    return classifier

def main():
    """Main funcation of the application."""
    print('Automated classfication of sentiment analysis datasets.')
    if len(sys.argv) != 4:
        print('Wrong parameter list. $1 = csv dataset; $2 = text fieldname; $3 category fieldname')
        quit()
    else:
        dataset, text_column, category_column = sys.argv[1], sys.argv[2], sys.argv[3]
        rows, classes_counter = treat_csv_dataset(dataset, text_column, category_column)
        majoritary_class = classes_counter.most_common()[0][1]/(sum(classes_counter.values()))
        rows_text, rows_label = generate_matrix(rows)
        x_raw = vectorize_data(rows_text)
        x_train, x_test, y_train, y_test = train_test_split(x_raw, rows_label, test_size=0.1, random_state=0)
        print("Majoritary class: {0}".format(majoritary_class))
        classifiers = ['SVM', 'NBM', 'C4.5']
        predictions = []
        for classifier in classifiers:
            predictions.append(classify_by_algorithm(classifier, x_test, y_test))
if __name__ == "__main__":
    main()
