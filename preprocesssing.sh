#filename="$1"
#filename="classificacao.csv"

filename="discover.arff"


rm -Rf output/* output_Maid/*
rm -Rf $filename

python pos_extractor.py
cd pretext
perl Start.pl
cd ..

echo -e "@relation sentiment \n"  >> $filename 
echo -e "@attribute classification {\"cfb\", \"noncfb\"}"  >> $filename 
#echo -e "@relation sentiment \n@classification {\"pos\", \"neg\"}\n"  >> $filename 
tail -n +2 pretext/discover/discover.names  |  sed 's/"//g' | sed 's/^/@attribute /g' | sed 's/:[^:]*$/ REAL/g'  >> $filename
echo -e "\n@data\n" >> $filename
sed -i '2d' $filename
sed s/...\output_Maid\.[0-9]*_//g pretext/discover/discover.data | sed s/,output_Maid//g >> $filename

