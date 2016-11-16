import sys, math, numpy, operator

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
		word_vecs[vector[0]] = map(float, vector[1:])
	return word_vecs

def l2norm(vector):
	return numpy.sqrt(numpy.sum(map(lambda x : x * x, vector)))

def get_word_lengths(word_vecs):
	return dict(map(lambda (k,v): (k, l2norm(v)), word_vecs.iteritems()))

def print_sorted(word_lengths):
	sorted_lengths = sorted(word_lengths.items(), key = operator.itemgetter(1))
	print("\n".join(map(str, sorted_lengths[0:20])))
        print("\n")
	print("\n".join(map(str, sorted_lengths[-20:-1])))
