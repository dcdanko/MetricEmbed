import sys, numpy, math, operator

# Creates an embedding of the input words into Euclidean space with
# dimension dims. The vectors are randomly generated s.t. their coordinates
# are N(0,1) distributed.
def rand_embedding(words,dims):
	word_vecs = {}
	for word in words:
		word_vecs[word] = numpy.randn(dims)
	return word_vecs

# Computes the l2norm of a vector
def l2norm(vector):
	return numpy.sqrt(numpy.sum(map(lambda x : x * x, vector)))
