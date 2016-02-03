filename="discover.arff"

rm -Rf ../dataset/output/* ../dataset/output_Maid/*

cd pretext
perl Start.pl
cd ..

rm -Rf $filename
echo -e "@relation sentiment \n"  >> $filename 
echo -e '@attribute classification {"cfb", "noncfb"}'  >> $filename 
tail -n +2 pretext/discover/discover.names  |  sed 's/"//g' | sed 's/^/@attribute /g' | sed 's/:[^:]*$/ REAL/g'  >> $filename
echo -e "\n@data\n" >> $filename
sed s/^.*output-g-2011_Maid\.[0-9]*_/\"/g pretext/discover/discover.data | sed s/,\\.\\.\\/dataset\\/output-g-2011_Maid//g  >> $filename 
