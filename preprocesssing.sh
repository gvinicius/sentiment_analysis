#filename="$1"
#filename="classificacao.csv"

filename="discover2.arff"

: << END
rm -Rf output/* output_Maid/*

python pos_extractor.py
cd pretext
perl Start.pl
cd ..

END
rm -Rf $filename
echo -e "@relation sentiment \n"  >> $filename 
echo -e "@attribute classification {\"neg\", \"pos\"}"  >> $filename 
#echo -e "@relation sentiment \n@classification {\"pos\", \"neg\"}\n"  >> $filename 
tail -n +2 pretext/discover/discover.names  |  sed 's/"//g' | sed 's/^/@attribute /g' | sed 's/:[^:]*$/ REAL/g'  >> $filename
echo -e "\n@data\n" >> $filename
#sed -i '2d' $filename
sed s/^.*output_Maid\.[0-9]*_/\"/g pretext/discover/discover.data | sed s/,\\.\\.\\/dataset\\/output_Maid//g  >> $filename 
#

