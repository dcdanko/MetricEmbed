#!/bin/bash

OUT=results.txt
SIZE=50
CORPUS=master_noP_noLine
METRIC=simpleMeasures.py

echo Creating Embeddings
#SWAPPEDFILES=$(ls *_swap*)
SWAPPEDFILES=$(ls text8_swap*_rep*)
#SWAPPEDFILES=(text8_swap11903646_rep0.txt text8_swap15304688_rep0.txt text8_swap17005208_rep1.txt text8_swap1700520_rep0.txt text8_swap20406250_rep1.txt text8_swap23807292_rep1.txt text8_swap27208334_rep1.txt text8_swap30609376_rep1.txt text8_swap32309896_rep2.txt text8_swap35710938_rep2.txt text8_swap39111980_rep2.txt text8_swap42513022_rep2.txt text8_swap45914064_rep2.txt text8_swap47614584_rep3.txt text8_swap51015626_rep3.txt text8_swap54416668_rep3.txt text8_swap57817710_rep3.txt text8_swap61218752_rep3.txt text8_swap62919272_rep4.txt text8_swap66320314_rep4.txt text8_swap69721356_rep4.txt text8_swap73122398_rep4.txt text8_swap76523440_rep4.txt text8_swap8502604_rep0.txt)
#SWAPPEDFILES=(text8_swap3401041 text8_swap510562 text8_swap6802083 text8_swap8502604)
for swapFile in $SWAPPEDFILES
do
echo 'swapping on file'
echo $swapFile
FOLDER=embeddings_$swapFile
echo $FOLDER
if [ ! -d $FOLDER ]; then
mkdir $FOLDER
fi
for i in {1..5}
do
    word2vec -train $swapFile -output $FOLDER/w2v$i -size $SIZE
		#echo 'train'
done
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
