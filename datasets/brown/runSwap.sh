#!/bin/bash

OUT=results.txt
SIZE=50
CORPUS=master_noP_noLine
METRIC=simpleMeasures.py

echo Creating Embeddings
#SWAPPEDFILES=$(ls *_swap*)
SWAPPEDFILES=$(ls master_no*_rep*)
#SWAPPEDFILES=(text8_swap3401041 text8_swap510562 text8_swap6802083 text8_swap8502604)
for swapFile in $SWAPPEDFILES
do
echo $swapFile
FOLDER=embeddings_$swapFile
echo $FOLDER
if [ ! -d $FOLDER ]; then
mkdir $FOLDER
fi
#for i in {1..5}
		i=1
    word2vec -train $swapFile -output $FOLDER/w2v$i -size $SIZE
#done
done

#echo \n Comparing Embeddings
#for i in {1..5}
#do
#    for (( j = i + 1 ; j <= 5 ; j++ ))
#    do
#        python $METRIC Embeddings/w2v$i Embeddings/w2v$j >> $OUT
#    done
#done

#echo \n Deleting Embeddings
#for i in {1..5}
#do
#    rm -rf Embeddings/w2v$i
#done
