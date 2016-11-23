import nltk, csv, numpy, sys, collections
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import treebank
from nltk.tokenize import RegexpTokenizer
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
# -*- coding: latin-1 -*-

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

def treat_csv_dataset(dataset, text_field, category_field, preprocessing_param):
    categories_counter = collections.Counter()
    with open(dataset, 'r', encoding="latin1") as csvfile:
        rows = []
        csv_reader = csv.reader(csvfile, quotechar='\"')
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row[text_field]
            category = str(row[category_field]).replace(" ","_").replace("/","").replace("'","")
            categories_counter[category] += 1
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(text)
            tagged = nltk.tag.pos_tag(raw_tokens)
            if preprocessing_param == 'tags':
                selected_tokens = [word for word,pos in tagged if pos in ['JJR','JJS','NNS','NNP'] ]
            elif preprocessing_param == 'no tags':
                selected_tokens = raw_tokens
            final_tokens = ""
            for word in selected_tokens:
                final_tokens += word + " " 
            #final_tokens = tokenizer.tokenize(final_tokens)
            rows.append((final_tokens, category))
        return rows, categories_counter

def generate_matrix(corpus):
    data = [c[0] for c in corpus]
    labels = [c[1] for c in corpus]
    return data, labels

def vectorize_data(texts):
    vectorizer = TfidfVectorizer(min_df=1)
    return vectorizer.fit_transform(texts)

def classify_by_algorithm(classifier_name, X_test, y_test):
    classifier = None
    if (classifier_name == 'SVM'):
        classifier = train_svm(X_test, y_test)
    elif(classifier_name == 'NBM'):
        classifier = MultinomialNB().fit(X_test, y_test)
    elif(classifier_name == 'C4.5'):
        classifier = tree.DecisionTreeClassifier().fit(X_test, y_test)
    pred = classifier.predict(X_test)
    print(classifier_name)
    print(confusion_matrix(pred, y_test))
    print(classifier.score(X_test, y_test))

def analyze(dataset, text_field, category_field, pos_condition):
    rows, classes_counter = treat_csv_dataset(dataset, text_field, category_field, pos_condition)
    majoritary_class = classes_counter.most_common()[0][1]/(sum(classes_counter.values()))
    rows_text, rows_label = generate_matrix(rows)
    X = vectorize_data(rows_text)
    X_train, X_test, y_train, y_test = train_test_split(X, rows_label, test_size=0.1, random_state=0)
    print(pos_condition)
    print(majoritary_class)
    classify_by_algorithm('SVM', X_test, y_test)
    classify_by_algorithm('NBM', X_test, y_test)
    classify_by_algorithm('C4.5', X_test, y_test)

def main():
    print ("Automated classfication of datasets for sentiment analysis.")
    expected_args_quantity = 4
    if len(sys.argv) != expected_args_quantity:
        print('Wrong arguments numbers')
        quit()
    else:
        dataset, text_field, category_field = sys.argv[1], sys.argv[2], sys.argv[3]
        analyze(dataset, text_field, category_field, 'tags')
        analyze(dataset, text_field, category_field, 'no tags')
if __name__ == "__main__":
    main()
