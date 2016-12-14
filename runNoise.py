import subprocess as sp
import embed_parse
import noiseCorpus as nc
from copy import copy, deepcopy
import numpy as np
import itertools as it

# filesList=(r'datasets/brown/master_noPunctuation',r'datasets/brown/master_noP_noLine', r'datasets/text8')
# filesList=(r'datasets/text8',)
filesList=(r'datasets/brown/master_noPunctuation',r'datasets/brown/master_noP_noLine')
for file in filesList:
	brownDocs=embed_parse.importToDoc(file)
	totalWords=sum(brownDocs.lineLength)
	numToSwap=np.arange(totalWords*0.1,totalWords,totalWords*0.2,dtype=int)
	totaln=0
	for rep in range(5):
		for n in np.diff(np.concatenate((np.array((0,)),numToSwap))):
		# for n in np.diff(np.concatenate((np.array((0,)),numToSwap))): # to reshuffle on each step
			# copyDoc=docs(brownDocs.lineLength,deepcopy(brownDocs.words)) # to reshuffle on each step
			nc.swapWords(brownDocs,n) # progressively swap same dataset
			totaln+=n
			onlyFile=file.split(sep=r'/')
			fileWrite=''.join((nc.standardSwapRename(onlyFile[-1], totaln),'_rep'+str(rep),'.txt'))
			fullFileWrite=r'/'.join(it.chain(onlyFile[:-1],(fileWrite,)))
			embed_parse.writeWordList(fullFileWrite, brownDocs.words)
			# runW2V=sp.run(['word2vec','-train',fileWrite,r'Embeddings/w2v_swap/'+onlyFile, '-size',50], check=True) # make a word2vec run
