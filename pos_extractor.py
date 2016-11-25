import nltk, csv, numpy, sys, collections
from nltk.corpus import stopwords
from nltk.corpus import treebank
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
# -*- coding: latin-1 -*-

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

# def treat_csv_dataset(dataset, text_field, category_field, preprocessing_param):
def treat_csv_dataset(dataset, text_field, category_field):
    classes_counter = collections.Counter()
    with open(dataset, 'r', encoding="latin1") as csvfile:
        rows = []
        csv_reader = csv.reader(csvfile, quotechar='\"')
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = row[text_field]
            classification = str(row[category_field]).replace(" ","_").replace("/","").replace("'","")
            classes_counter[classification] += 1
            tokenizer = RegexpTokenizer(r'\w+') 
            raw_tokens = tokenizer.tokenize(text)
            tagged = nltk.tag.pos_tag(raw_tokens)
            final_tokens = ""
            for word in raw_tokens:
                final_tokens += word + " " 
            rows.append((final_tokens, classification))
        return rows, classes_counter

def generate_matrix(corpus):
    data = [c[0] for c in corpus]
    labels = [c[1] for c in corpus]
    return data, labels

def vectorize_data(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

def classify_by_algorithm(classifier_name, X_situation, y_situation):
    classifier = None
    if (classifier_name == 'SVM'):
        classifier = train_svm(X_situation, y_situation)
    elif(classifier_name == 'NBM'):
        classifier = MultinomialNB().fit(X_situation, y_situation)
    elif(classifier_name == 'C4.5'):
        classifier = tree.DecisionTreeClassifier().fit(X_situation, y_situation)
    prediction = classifier.predict(X_situation)
    print(classifier_name)
    print(confusion_matrix(prediction, y_situation))
    print(classification_report(prediction, y_situation))
    print(classifier.score(X_situation, y_situation))

def main():
    print ('Automated classfication of sentiment analysis datasets.')
    if len(sys.argv) != 4:
        print('Wrong number of parameters. The correct is: $1 = dataset csv file; $2 = text fieldname in csv; $3 category fieldname in csv')
        quit() 
    else:
        dataset, text_column, category_column = sys.argv[1], sys.argv[2], sys.argv[3]
        rows, classes_counter = treat_csv_dataset(dataset, text_column, category_column)
        majoritary_class = classes_counter.most_common()[0][1]/(sum(classes_counter.values()))
        rows_text, rows_label = generate_matrix(rows)
        X = vectorize_data(rows_text)
        X_train, X_test, y_train, y_test = train_test_split(X, rows_label, test_size=0.1)
        print("Majoritary class: {0}".format(majoritary_class))
        classifiers = ['SVM', 'NBM', 'C4.5']
        for classifier in classifiers:
            classify_by_algorithm(classifier,  X_test, y_test)
if __name__ == "__main__":
    main()
