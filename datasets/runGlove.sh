#!/bin/bash
#FILE_PATH=~/Documents/AdvTopicModelling/Project/code/MetricEmbed/datasets
#CORPUS_LIST=$(find $FILE_PATH -name 'text8_swap*')

# FILE_PATH=~/Documents/AdvTopicModelling/project/code/MetricEmbed/datasets
# CORPUS_LIST=$(find $FILE_PATH -name 'text8_swap*')
FILE_PATH=~/Documents/AdvTopicModelling/project/code/MetricEmbed/datasets/brown
CORPUS_LIST=$(find $FILE_PATH -name 'master_noPunctuation_swap*')
echo $CORPUS_LIST
for CORPUS in $CORPUS_LIST
do
	echo $CORPUS
	#the following is cut and paste from demo_noCorpus. Sourcing was doing funny things
	#. demo_noCorpus.sh
		make
		if [ ! -e text8 ]; then
		  if hash wget 2>/dev/null; then
		    wget http://mattmahoney.net/dc/text8.zip
		  else
		    curl -O http://mattmahoney.net/dc/text8.zip
		  fi
		  unzip text8.zip
		  rm text8.zip
		fi
		
		VOCAB_FILE=vocab.txt
		COOCCURRENCE_FILE=cooccurrence.bin
		COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
		BUILDDIR=build
		SAVE_FILE=vectors
		VERBOSE=2
		MEMORY=4.0
		VOCAB_MIN_COUNT=5
		VECTOR_SIZE=50
		MAX_ITER=15
		WINDOW_SIZE=15
		BINARY=2
		NUM_THREADS=8
		X_MAX=10
		
		$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
		if [[ $? -eq 0 ]]
		  then
		  $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
		  if [[ $? -eq 0 ]]
		  then
		    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
		    if [[ $? -eq 0 ]]
		    then
		       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
		       if [[ $? -eq 0 ]]
		       then
		           if [ "$1" = 'matlab' ]; then
		               matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
		           elif [ "$1" = 'octave' ]; then
		               octave < ./eval/octave/read_and_evaluate_octave.m 1>&2 
		           else
		               python eval/python/evaluate.py
		           fi
		       fi
		    fi
		  fi
		fi
	# end copy-paste
	echo "\n===========run completed=============="
	CORPUS_NAME=${CORPUS##*/}
	echo 'copying '$CORPUS_NAME
	cp vectors.txt $FILE_PATH/gloveEmbed_$CORPUS_NAME
done
