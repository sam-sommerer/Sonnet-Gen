mkdir data/conceptnet

cd data/conceptnet

curl https://home.ttic.edu/~kgimpel/comsense_resources/train100k.txt.gz -o train100k.txt.gz
curl https://home.ttic.edu/~kgimpel/comsense_resources/dev1.txt.gz -o dev1.txt.gz
curl https://home.ttic.edu/~kgimpel/comsense_resources/dev2.txt.gz -o dev2.txt.gz
curl https://home.ttic.edu/~kgimpel/comsense_resources/test.txt.gz -o test.txt.gz

gunzip train100k.txt.gz
gunzip dev1.txt.gz
gunzip dev2.txt.gz
gunzip test.txt.gz

cd ..