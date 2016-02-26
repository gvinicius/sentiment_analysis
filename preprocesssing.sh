filename="discover.arff"

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
#sed -i 's/^.*output-g-'"$1"'\\_Maid\.[0-9]*_/\"/g' pretext/discover/discover.data 
#sed -i 's/,\\.\\.\\/dataset\\/output-g-'"$1"'\\_Maid//g' pretext/discover/discover.data 
#cat pretext/discover/discover.data >> $filename 
bash patterns_extrator.sh
git checkout -- pretext/config.xml
