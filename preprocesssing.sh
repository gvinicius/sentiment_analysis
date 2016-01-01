#filename="$1"
#filename="classificacao.csv"

filename="discover.csv"


rm -Rf output/* output_Maid/*
rm -Rf $filename

python pos_extractor.py
cd pretext
perl Start.pl
cd ..

echo "classification," | tr '\n' ' '  >> discover.csv 
tail -n +2 pretext/discover/discover.names  |  sed 's/"//g' | sed s/:[^:]*$/,/g  | sed '$ s/.$//' | tr '\n' ' '  >> discover.csv
echo -e "\n" >> discover.csv
sed -i '2d' discover.csv
sed s/...\output_Maid\.[0-9]*_//g pretext/discover/discover.data | sed s/,output_Maid//g >> discover.csv

