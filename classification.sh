#filename="$1"
#filename="classificacao.csv"

filename="discover.csv"

result_filename="result.txt"

rm -Rf output/* output_Maid/*
rm -Rf $result_filename
rm -Rf $filename

python pos_extractor.py
cd pretext
perl Start.pl
cd ..

echo "classification," | tr '\n' ' '  >> discover.csv 
tail -n +2 pretext/discover/discover.names  |  sed 's/"//g' | sed s/:[^:]*$/,/g  | sed '$ s/.$//' | tr '\n' ' '  >> discover.csv
echo -e "\n" >> discover.csv
sed '2d' discover.csv
sed s/...\output_Maid\.[0-9]*_//g pretext/discover/discover.data | sed s/,output_Maid//g >> discover.csv



echo "Beginning the classification"

echo "NaiveBayes" >> $result_filename
java -cp $CLASSPATH weka.classifiers.bayes.NaiveBayes -c 1 -o -t "discover.csv" >>  $result_filename
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
