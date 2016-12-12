#!/bin/python
import argparse
import embed_parse
import numpy as np

def buildArgs():
	parser = argparse.ArgumentParser(description='A default script')
	parser.add_argument('--sep',dest='sep',type=str,default=r' ',help='Seperator for embedding tab;e')
	parser.add_argument('-i','--input',dest='inputDoc',type=str, help='File containing the word embedding')
	parser.add_argument('-n','--nswap', dest='numSwap',type=int,default=1000,help='number of tiems to swap words at random')
	parser.add_argument('-o','--output',dest='outputDoc',type=str,default=None,help='file to output noise level')

	args = parser.parse_args()
	if args.inputDoc is None:
		raise FileNotFoundError('failed to give input file for noiseCorpus.py')
	return args

def swapWords(docs, numToSwap):
	words=docs.words
	lineToSwapFrom=np.random.choice(len(words),size=numToSwap, p=np.array(docs.lineLength)/sum(docs.lineLength))
	lineToSwapWith=np.random.choice(len(words),size=numToSwap, p=np.array(docs.lineLength)/sum(docs.lineLength))
	for lineFromIndx, lineToIndx in zip(lineToSwapFrom,lineToSwapWith):
		wordToSwapFrom=np.random.randint(0,len(words[lineFromIndx]))
		wordToSwapWith=np.random.randint(0,len(words[lineToIndx]))
		tmp=words[lineFromIndx][wordToSwapFrom]
		words[lineFromIndx][wordToSwapFrom]=words[lineToIndx][wordToSwapWith]
		words[lineToIndx][wordToSwapWith]=tmp

def standardSwapRename(infilename, numSwap):
	breakStr=infilename.split('.')
	if len(breakStr)>1: # insert '_swap#' before file extension period.
		partialOutput=".".join(breakStr[:-1])
		outfilename=("_swap"+str(numSwap)).join((partialOutput,breakStr[-1]))
		return outfilename
	else: # assume no file extension
		return infilename+'_swap'+str(numSwap)
def importAndSwap(infilename,numSwap=None, outfilename=None, sep=" "):
	docs=embed_parse.importToDoc(infilename, sep)
	if numSwap is None:
		numSwap=sum(docs.lineLength)*0.1
	swapWords(docs,numSwap)
	if outfilename is None:
		outfilename=standardSwapRename(infilename,numSwap)
	embed_parse.writeWordList(outfilename, docs.words, args.sep)

if __name__=="__main__":
	args=buildArgs()
	importAndSwap(args.inputDoc,args.numSwap,args.outputDoc, args.sep)
