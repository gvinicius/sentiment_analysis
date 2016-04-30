filename="discover.arff"

python pos_extractor.py $1 $2
cd pretext
sed -i 's/ANNO/'"$1"'/g' config.xml
perl Start.pl
cd ..

rm -Rf $filename -v
echo -e "@relation sentiment \n"  >> $filename 
echo -e '@attribute classification {"cfb", "noncfb"}'  >> $filename 
tail -n +2 pretext/discover/discover.names  |  sed 's/"//g' | sed 's/^/@attribute /g' | sed 's/:[^:]*$/ REAL/g'  >> $filename
echo -e "\n@data\n" >> $filename
a="s/^\".*$1\_Maid\/[0-9]*\_/\"/g"
b="s/,\".*$1\_Maid\/[0-9]*\_/\/g"

sed -e "s/^\".*$1\_Maid\/[0-9]*\_/\"/g" pretext/discover/discover.data | sed -e "s/,\\.\\.\\/dataset\\/output-g-$1\\_Maid//g" >> $filename 

bash patterns_extrator.sh
sed -i 's/'"$1"'/ANNO/g' pretext/config.xml
#git checkout -- pretext/config.xml
