#filename="$1"
#filename="classificacao.csv"


rm -Rf /output/* /pretext/textos/* /pretext/textos_Maid

python pos_extractor.py
perl pretext/Start.pl

echo "\"classification\"," >> discover.csv
# sed s/:[^:]*$/,/g pretext/discover/discover.names | tr '\n' ' ' >> discover.csv
# sed s/__[^__]*$//g pretext/discover/discover.names | sed s/:[^:]*$/,/g | tr '\n' ' ' >> discover.csv

filename="discover.csv"

$result_filename = result.txt

rm -Rf $result_filename

echo "Beginning the classification"

echo "NaiveBayes" >> $result_filename
java -cp $CLASSPATH weka.classifiers.bayes.NaiveBayes -c 1 -o -t $filename >>  $result_filename
echo "--- end of NaiveBayes ---" >> $result_filename

echo "NaiveBayesMultinomial" >> $result_filename
java -cp $CLASSPATH weka.classifiers.bayes.NaiveBayesMultinomial -c 1 -o -t $filename >>  $result_filename
echo "--- end of NaiveBayesMultinominal ---" >> $result_filename

echo "SVM" >> $result_filename
java -cp $CLASSPATH weka.classifiers.functions.SMO -c 1 -o -t $filename >>  $result_filename
echo "--- end of SVM ---" >> $result_filename

echo "KNN" >> $result_filename
#java -cp $CLASSPATH  weka.core.neighboursearch.LinearNNSearch -c 1 -o -t $filename >>  $result_filename
java -cp $CLASSPATH  weka.classifiers.lazy.IBk -c 1 -o -t $filename >>  $result_filename
echo "--- end of KNN ---" >> $result_filename

echo "J48" >> $result_filename
java -cp $CLASSPATH weka.classifiers.trees.J48 -c 1 -o -t $filename >>  $result_filename
echo "--- end of J48  ---" >> $result_filename

echo "Classification ended and the output is set in $result_filename"
