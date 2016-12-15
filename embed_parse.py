import numpy as np
from collections import namedtuple
import os
from parse import parse as pyparse
import os.path
import pandas as pd

""" script for reading in the embedding from word2vec and GLoVe

key outputs:
word_vecs -- a dictionary of form "word" -> embedLocation
word_lengths --  a dictionary of the form "word" -> l2norm
print-sorted -- a function that prints the top 20 and bottom 20 word_lengths according to the word_lengths
"""

class RichEmbedding:
        def __init__(self,tool,ndim,corpus,replicate,embedding):
                self.tool = tool
                self.ndim = ndim
                self.corpus = corpus
                self.replicate = replicate
                self.embedding = embedding

def parseEmbeddingName(name):
        '''
        Takes an embedding name in the form of <tool>-dim=<N>-corpus=<corpus>-<replicate>.txt
        and outputs (<tool>,<N>,<corpus>,<replicate>)
        '''
        name = os.path.basename(name)
        m = pyparse('{tool}-dim={N}-corpus={corpus}-{replicate}.txt',name).named
        return (m['tool'], m['N'], m['corpus'], m['replicate'])

def parseEmbeddingWithMetadata(filename):
        df = pd.DataFrame.from_dict(parse(filename),orient='index')
        df.fillna(value=0,inplace=True)
#        df = pd.DataFrame.from_dict(parse(filename))
#        df = pd.DataFrame.from_csv(filename,sep=' ',header=None)
        tool,ndim,corpus,rep = parseEmbeddingName(filename)
        return RichEmbedding(tool,ndim,corpus,rep,df)

def parse(filename):
	word_vecs = {}
	bulkLines=False
	for line in open(filename):
		vector = line.split()
		if bulkLines or len(vector)>2:
			word_vecs[vector[0]] = list(map(float, vector[1:]))
		bulkLines=True
	return word_vecs

embedding=namedtuple('embedding', 'dict evect vectkeys')
def parseToEmbedding(filename):
	"""
	improved parse. returns an embedding object instead of a dictionary
	:param filename:
	:return:
	"""
	dict1 = parse(filename)
	sorted_vocab = sorted(dict1.keys())
	embed1 = np.array(list(map(lambda word: dict1[word], sorted_vocab)))
	return embed1

def pairDictsToPairEmbed(dict1,dict2):
	sorted_vocab = sorted(dict1.keys() & dict2.keys())
	embed1 = np.array(list(map(lambda word: dict1[word], sorted_vocab)))
	embed2 = np.array(list(map(lambda word: dict2[word], sorted_vocab)))
	return (embedding( dict1, embed1, sorted_vocab), embedding(dict2, embed2, sorted_vocab))

def importTwo(filename1, filename2):
	"""
	imports two files as embeddings,
	orders the words by the 1st file and then returns two embedding objects,
	the 1st corresponding to the 1st file and the 2nd correesponding to the second

	sorts so that the embedding vectors are the same ordering and have the same words (intersection of words)
	"""
	dict1 = parse(filename1)
	dict2 = parse(filename2)
	return pairDictsToPairEmbed(dict1,dict2)

def importWordList(filename, sep=' '):
	"""
	imports documents into a list of lists. Each list in the top level list is a "document" as defined by the presence of a new line.
	Each element of the lower level list is a word, as defined by th seperator function
	strips newline characters from words.
	:param filename:
	:param sep: the seperator function. If you insert a string (str) will use str.split(sep). Otherwise use a function that takes in one argument (the line) and returns an array of words
	:return:
	"""
	with open(filename) as f:
		lineList=f.readlines()
	if isinstance(sep, str):
		wordList=[line.split(sep) for line in lineList]
	else:
		wordList=[sep(line) for line in lineList]
	stripNewlines(wordList)
	return wordList

def generateBrownFilesList():
	"""
	generates a list of all brown corpus embeddings
	"""
	saveFile='brownFilesToUse.txt'
	if os.path.isfile(saveFile):
		print('loading saved brown files list')
		with open(saveFile) as f:
			lines=f.readlines()
		return list(filter(lambda s: len(s)>0, map(lambda l: l.rstrip('\n'),lines)))
	else:
		files=ls_fullpath('Embeddings')
		embeddingsFolderFiles=list(filter(lambda s: s.find('corpus='+'brown') != -1 and s.find('dim='+'50') != -1, files))
		files=ls_fullpath(r'datasets/brown')
		gloveEmbedNoise=list(filter(lambda s: s.find('gloveEmbed_master_noPunctuation') != -1, files))
		w2vEmbedNoise=[]
		for w2vFam in filter(lambda s: s.find('embeddings_master_noPunctuation') != -1, files):
			for file in ls_fullpath(w2vFam):
				w2vEmbedNoise.append(file)
		return w2vEmbedNoise+gloveEmbedNoise+embeddingsFolderFiles
def breakFileListEmbedAlgo(fileList):
	"""
	:param fileList:
	:return: tuple (w2v files, glove files)
	"""
	return (list(filter(lambda s: s.find('w2v') != -1, fileList)), list(filter(lambda s: s.find('glove') != -1 or s.find('GloVe')!=-1, fileList)))
def fileListNoSwap(fileList):
	return list(filter(lambda s: s.find('swap') == -1, fileList))

def generateText8FilesList():
	"""
	generates a list of all text8 corpus embeddings
	"""
	saveFile='text8FilesToUse.txt'
	if os.path.isfile(saveFile):
		with open(saveFile) as f:
			lines=f.readlines()
		return list(filter(lambda s: len(s)>0, map(lambda l: l.rstrip('\n'),lines)))
	else:
		files=ls_fullpath('Embeddings')
		embeddingsFolderFiles=list(filter(lambda s: s.find('corpus='+'text8') != -1 and s.find('dim='+'50') != -1, files))
		files=ls_fullpath(r'datasets')
		gloveEmbedNoise=list(filter(lambda s: s.find('gloveEmbed_text8') != -1, files))
		w2vEmbedNoise=[]
		for w2vFam in filter(lambda s: s.find('embeddings_text8') != -1, files):
			for file in w2vFam:
				w2vEmbedNoise.append(file)
		return w2vEmbedNoise+gloveEmbedNoise+embeddingsFolderFiles

def stripNewlines(wordList):
	for line in wordList:
		for indx in range(len(line)):
			word=line[indx]
			line[indx]=word.rstrip('\n\r')

def writeWordList(filename, wordList, sep=' '):
	"""
	performs the inverse of importDocs, takes in the filename to write and the docs. Merges the lower-level array with sep and writes to file.
	:param filename:
	:param sep:
	:return: nothing
	"""
	if isinstance(sep, str):
		merger=sep.join
	else:
		merger=sep
	mergeAndNewline=lambda line: merger(line)+'\n'
	with open(filename, 'w') as f:
		f.writelines(map(mergeAndNewline,wordList))
docs=namedtuple('docs','lineLength words')
def toDoc(wordList):
	lineLength=[len(line) for line in wordList]
	return docs(lineLength, wordList)
def importToDoc(fileName, sep=' '):
	return toDoc(importWordList(fileName,sep))

def ls_fullpath(dir):
	return [os.path.join(dir,f) for f in os.listdir(dir)]

def l2norm(vector):
	return np.sqrt(np.sum(list(map(lambda x : x * x, vector))))

def get_word_lengths(word_vecs):
	return dict(map(lambda k,v: (k, l2norm(v)), word_vecs.iteritems()))

# def print_sorted(word_lengths):
#     sorted_lengths = sorted(word_lengths.items(), key = operator.itemgetter(1))
#     print("\n".join(map(str, sorted_lengths[0:20])))
#     print("\n")
#     print("\n".join(map(str, sorted_lengths[-20:-1])))
