"""This module does automated text classification ensuring AB tests for preprocessing techniques."""
import sys
import csv
import collections
import nltk as nt
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from scipy import stats
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import TruncatedSVD 
from sklearn.decomposition import FactorAnalysis as LDA

def train_svm(x_data, y_data):
    """ Create and train a Support Vector Machine. """
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
            final_tokens = ""
            for word in raw_tokens:
                if word not in stopwords.words('english'):
                    final_tokens += word + " "
            rows.append((final_tokens, classification))
        return rows, classes_counter

def generate_matrix(corpus):
    """This function prepares the attribute-value matrices."""
    data = [c[0] for c in corpus]
    labels = [c[1] for c in corpus]
    return data, labels

def vectorize_data(texts, condition):
    """This function vectorizes text to matrices."""
    # vectorizer = TfidfVectorizer()
    #vectorizer = TfidfVectorizer(max_features=111)
    #vectorizer = TfidfVectorizer(max_features=111)
    vectorizer = TfidfVectorizer()
    transformation = vectorizer.fit_transform(texts)
    if condition == 'pca':
        lda = LDA()
        transformation = lda.fit_transform(transformation.toarray())
        # r = robjects.r
        # pca = r.princomp(transformation)
        # transformation =  pca.rotation
    #dimensionality_notion = len(transformation.toarray()[0])
    dimensionality_notion = 1
    return transformation, dimensionality_notion

def classify_by_algorithm(classifier_name, x_test, y_test, kfold):
    """This function enables classification by a series of algorithms and train/test situation."""
    if classifier_name == 'SVM':
        classifier = train_svm(x_test, y_test)
    elif classifier_name == 'NBM':
        classifier = MultinomialNB().fit(x_test, y_test)
    elif classifier_name == 'CART':
        classifier = DecisionTreeClassifier(random_state=0).fit(x_test, y_test)
    print(classifier_name)
    cross_result = cross_val_score(classifier, x_test, y_test, cv=kfold)
    accuracy = cross_result.mean()*100
    standard_deviation = cross_result.std()
    print("Acc: {0} %".format(accuracy))
    print("Std: {0}".format(standard_deviation))
    return accuracy, cross_result

def main():
    """Main method of the application."""
    print('Automated classfication of sentiment analysis datasets.')
    if len(sys.argv) != 4:
        print('Wrong parameter list. $1 = csv dataset; $2 = text fieldname; $3 category fieldname')
        quit()
    else:
        dataset, text_column, category_column = sys.argv[1], sys.argv[2], sys.argv[3]
        print(dataset)
        matrix_conditions = ['tdidf', 'pca']
        classifiers = ['SVM', 'CART']
        result_labels = ['SVM', 'CART']
        result_labels.append('dim')
        runs_set = []
        results_set = []
        with open('results.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow([dataset])
            csvwriter.writerow(result_labels)
        for matrix in matrix_conditions:
            results = []
            rows, classes_counter = treat_csv_dataset(dataset, text_column, category_column)
            majority_class = classes_counter.most_common()[0][1]/(sum(classes_counter.values()))
            with open('results.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([round(majority_class,2)])
            rows_text, labels = generate_matrix(rows)
            x_raw, dim = vectorize_data(rows_text, matrix)
            print("Majority class: {0}".format(majority_class))
            x_train, x_test, y_train, y_test = train_test_split(x_raw, labels,
                                                                  test_size=0.1, random_state=0)
            kfold = StratifiedKFold(n_splits=10, shuffle=False)
            with open('results.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                results = []
                csv_results = []
                for classifier in classifiers:
                    accuracy, runs = classify_by_algorithm(classifier, x_test, y_test, kfold)
                    results.append(accuracy)
                    runs_set.append(runs)
                results_set.extend(results)
                csv_results.extend(results)
                csv_results.append(dim)
                csvwriter.writerow(csv_results)
        """ Statistical test to check tagging technique impact. """
        """
        elements_notag = [results_set[0], results_set[1], results_set[2]]
        elements_tag = [results_set[3], results_set[4], results_set[5]]
        h_statistic, p_value = stats.mannwhitneyu(elements_notag, elements_tag)
        print("H-statistic:", h_statistic)
        print("P-Value:", p_value)
        with open('results.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow([round(p_value,2)])
        """
if __name__ == "__main__":
    main()
