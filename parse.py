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
	contents = open(filename)
	line = next(contents)
	vector = line.split()
	if len(vector) != 2:
		word_vecs[vector[0]] = list(map(float, vector[1:]))
	for line in contents:
		vector = line.split()
		word_vecs[vector[0]] = list(map(float, vector[1:]))
	return word_vecs

embedding=namedtuple('embedding', 'dict evect vectkeys')
def importTwo(filename1, filename2):
	"""
	imports two files as embeddings,
	orders the words by the 1st file and then returns two embedding objects, the 1st corresponding to the 1st file and the 2nd correesponding to the second"""
	dict1 = parse(filename1)
	dict2 = parse(filename2)
	sorted_vocab = sorted(dict1)
	embed1 = np.array(list(map(lambda word: dict1[word], sorted_vocab)))
	embed2 = np.array(list(map(lambda word: dict2[word], sorted_vocab)))
	return (embedding(dict1, embed1, sorted_vocab), embedding(dict2, embed2, sorted_vocab))

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
