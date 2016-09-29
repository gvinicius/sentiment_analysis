filename="discover3.arff"

python3 pos_extractor.py
cd pretext
perl Start.pl
cd ..

rm -Rf $filename -v
echo "@relation sentiment "  >> $filename 
tail -n +2 pretext/discover/discover.names  |  sed 's/"//g' | sed 's/^/@attribute /g' | sed 's/:[^:]*$/ REAL/g'  >> $filename
echo '@attribute classification_name {"Neutral","Positive","Negative","Not_Related"}'  >> $filename 
#echo '@attribute classification_name {"FOR","NEUTRAL","AGAINST","NONE_OF_THE_ABOVE"}'  >> $filename 
echo "@data" >> $filename

# working on terminal
sed "s/\(...\\..\/dataset\/economics\/result\_Maid..[0-9]*\_\)//" pretext/discover/discover.data  | sed "s/\.txt\.swp//" | sed 's/,\.\.\/dataset\/economics\/result\_Maid//' | awk -F','  ' BEGIN { OFS = ","; ORS = "\n" } { t = $1; $1 = $NF; $NF = t; print; } ' >> discover3.arff

#sed -e "s/^\\/result\\_Maid\/[0-9]*\_/\"/g" pretext/discover/discover.data | sed -e "s/,\\.\\.\\/dataset\\/economics\\/result/\\_Maid//g" >> $filename 

# sed "s/\(...\\..\/dataset\/economics\/result\_Maid..[0-9]*\_\)//" testing])
# sed "s/\(...\\..\/dataset\/economics\/result\_Maid..[0-9]*\_\)//"  | sed "s/\.txt\.swp//" | sed 's/,\.\/dataset\/economics\/result\_Maid//' >> discover3.arff

# bash patterns_extrator.sh
# sed -i 's/'"$1"'/ANNO/g' pretext/config.xml
#git checkout -- pretext/config.xml
