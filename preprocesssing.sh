filename="discover.arff"

# python pos_extractor.py
cd pretext
perl Start.pl
cd ..

rm -Rf $filename -v
echo -e "@relation sentiment \n"  >> $filename 
echo -e '@attribute classification {"yes", "no"}'  >> $filename 
tail -n +2 pretext/discover/discover.names  |  sed 's/"//g' | sed 's/^/@attribute /g' | sed 's/:[^:]*$/ REAL/g'  >> $filename
echo -e "\n@data\n" >> $filename

#sed -e "s/^\\/result\\_Maid\/[0-9]*\_/\"/g" pretext/discover/discover.data | sed -e "s/,\\.\\.\\/dataset\\/economics\\/result/\\_Maid//g" >> $filename 

# bash patterns_extrator.sh
# sed -i 's/'"$1"'/ANNO/g' pretext/config.xml
#git checkout -- pretext/config.xml
