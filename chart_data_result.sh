chart_dataset=comparaison.dat
a=0
rm -Rf $chart_dataset

echo -e "$a \t NaiveBayes" | tr -s "\n" "\t" >>  $chart_dataset
a=$((a+1))
grep ^NaiveBayes -A 40 result.txt | awk '/^Correctly/{i++}i==2' | grep -o '.\{9\}$' | head -1 | tr "%" " " >>  $chart_dataset
echo -e "$a \t NaiveBayesMultinomial" | tr -s "\n" "\t" >>  $chart_dataset
a=$((a+1))
grep ^NaiveBayesMultinomial -A 40 result.txt | awk '/^Correctly/{i++}i==2' | grep -o '.\{9\}$' | head -1 | tr "%" " " >>  $chart_dataset
echo -e "$a \t SVM" | tr -s "\n" "\t" >>  $chart_dataset
a=$((a+1))
grep ^SVM -A 40 result.txt | awk '/^Correctly/{i++}i==2' | grep -o '.\{9\}$' | head -1 | tr "%" " " >>  $chart_dataset
echo -e "$a \t KNN" | tr -s "\n" "\t" >>  $chart_dataset
a=$((a+1))
grep ^KNN -A 40 result.txt | awk '/^Correctly/{i++}i==2' | grep -o '.\{9\}$' | head -1 | tr "%" " " >>  $chart_dataset
echo -e "$a \t J48" | tr -s "\n" "\t"  >>  $chart_dataset
a=$((a+1))
grep ^J48 -A 40 result.txt | awk '/^Correctly/{i++}i==2' | grep -o '.\{9\}$' | head -1 | tr "%" " " >>  $chart_dataset

gnuplot -e set boxwidth 0.5; set style fill solid; plot "$filename" using 1:3:xtic(2) with boxes;
