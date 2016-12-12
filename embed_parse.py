import numpy as np
from collections import namedtuple
import os

""" script for reading in the embedding from word2vec and GLoVe

key outputs:
word_vecs -- a dictionary of form "word" -> embedLocation
word_lengths --  a dictionary of the form "word" -> l2norm
print-sorted -- a function that prints the top 20 and bottom 20 word_lengths according to the word_lengths
"""

def parse(filename):
	word_vecs = {}
	for line in open(filename):
		vector = line.split()
		word_vecs[vector[0]] = list(map(float, vector[1:]))
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

def importTwo(filename1, filename2):
	"""
	imports two files as embeddings,
	orders the words by the 1st file and then returns two embedding objects,
	the 1st corresponding to the 1st file and the 2nd correesponding to the second

	sorts so that the embedding vectors are the same ordering and have the same words (intersection of words)
	"""
	dict1 = parse(filename1)
	dict2 = parse(filename2)
	sorted_vocab = sorted(dict1.keys() & dict2.keys())
	embed1 = np.array(list(map(lambda word: dict1[word], sorted_vocab)))
	embed2 = np.array(list(map(lambda word: dict2[word], sorted_vocab)))
	return (embedding(dict1, embed1, sorted_vocab), embedding(dict2, embed2, sorted_vocab))

def importWordList(filename, sep=' '):
	"""
	imports documents into a list of lists. Each list in the top level list is a "document" as defined by the presence of a new line.
	Each element of the lower level list is a word, as defined by th seperator function
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
	return wordList
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
	with open(filename, 'w') as f:
		for line in wordList:
			f.write(merger(line))
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
