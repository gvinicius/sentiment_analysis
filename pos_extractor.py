"""This module does automated text classification ensuring AB tests for preprocessing techniques."""
import sys
import csv
import collections
import nltk as nt
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

def classify_by_algorithm(classifier_name, x_original, rows_label):
    """This function enables classification by a series of algorithms and train/test situation."""
    x_train, x_test, y_train, y_test = train_test_split(x_original, rows_label, test_size=0.1, random_state=0)
    if classifier_name == 'SVM':
        classifier = train_svm(x_test, y_test)
    elif classifier_name == 'NBM':
        classifier = MultinomialNB().fit(x_test, y_test)
    elif classifier_name == 'C4.5':
        classifier = DecisionTreeClassifier().fit(x_test, y_test)
    prediction = classifier.predict(x_test)
    print(classifier_name)
    # print(confusion_matrix(prediction, y_test))
    # print(classification_report(prediction, y_test))
    print("Score:{0}".format(classifier.score(x_test, y_test)))
    kfold = StratifiedKFold(n_splits=10, shuffle = False)
    cross_result = cross_val_score(classifier, x_test, y_test, cv=kfold)
    accuracy = cross_result.mean()
    standard_deviation = cross_result.std()
    print("Acc:{0}".format(accuracy))
    print("Std:{0}".format(standard_deviation))
    return (classifier_name, cross_result)

def validate_stat(predictions_pair, test_name):
    """This method implements t student comparison of two classifiers runnings."""
    print("{0} x {1}".format(predictions_pair[0][0], predictions_pair[1][0]))
    if test_name == 'student_t':
        paired_sample = stats.ttest_rel(predictions_pair[0][1], predictions_pair[1][1])
    elif test_name == 'wilcoxon':
        paired_sample = stats.ranksums(predictions_pair[0][1], predictions_pair[1][1])
    p_value = paired_sample[1]
    if (p_value < 0.05):
        if predictions_pair[0][1].mean() > predictions_pair[1][1].mean():
            best_classifier = predictions_pair[0]
        elif predictions_pair[0][1].mean() < predictions_pair[1][1].mean():
            best_classifier = predictions_pair[1]
        else:
            print('There is no significant difference between the two algorithms performance.')
            return [0, 'none']
        print('The best classifier is: {0}.'.format(best_classifier[0]))
        return [best_classifier[1].mean(), best_classifier[0]]
    else:
        print('There is no significant difference between the two algorithms performance.')
        return [0, 'none']
        
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
        for classifier in classifiers:
            predictions.append(classify_by_algorithm(classifier, x_raw, row_labels))
        test_name = 'wilcoxon'
        test_results = dict()
        r1 = validate_stat([predictions[0],predictions[1]], test_name)
        r2 = validate_stat([predictions[0],predictions[2]], test_name)
        r3 = validate_stat([predictions[1],predictions[2]], test_name)
        test_results[r1[0]] = r1[1]
        test_results[r2[0]] = r2[1]
        test_results[r3[0]] = r3[1]
        best_score = max(test_results.keys())
        print(best_score)
        if best_score != 0:
            best = test_results[best_score]
            print("After the statistical test, the best classifier is: {0}".format(best))
        else:
            print("After the statistical test, there is no difference between classifiers.")
if __name__ == "__main__":
    main()
