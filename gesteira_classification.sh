filename="$1"
rm -Rf result.txt
echo "Begining the classification"

echo "NaiveBayes" >> result.txt
java -cp $CLASSPATH weka.classifiers.bayes.NaiveBayes -o -t $filename >>  result.txt
echo "--- end of NaiveBayes ---" >> result.txt

echo "NaiveBayesMultinomial" >> result.txt
java -cp $CLASSPATH weka.classifiers.bayes.NaiveBayesMultinomial -o -t $filename >>  result.txt
echo "--- end of NaiveBayesMultinominal ---" >> result.txt

echo "SVM" >> result.txt
java -cp $CLASSPATH weka.classifiers.functions.SMO -o -t $filename >>  result.txt
echo "--- end of SVM ---" >> result.txt

echo "KNN" >> result.txt
#java -cp $CLASSPATH  weka.core.neighboursearch.LinearNNSearch -o -t $filename >>  result.txt
java -cp $CLASSPATH  weka.classifiers.lazy.IBk -o -t $filename >>  result.txt
echo "--- end of KNN ---" >> result.txt

echo "J48" >> result.txt
java -cp $CLASSPATH weka.classifiers.trees.J48 -o -t $filename >>  result.txt
echo "--- end of J48  ---" >> result.txt

echo "Classification ended and the output is set in result.txt"
