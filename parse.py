import sys, math, numpy, operator

embedding = open(sys.argv[1])

word_vecs = {}
total_words = 0

for line in embedding:
	vector = line.split()
	total_words += 1
	word_vecs[vector[0]] = map(lambda x: float(x), vector[1:])

def l2norm(vector):
	return numpy.sqrt(numpy.sum(map(lambda x : x * x, vector)))

word_lengths = dict(map(lambda (k,v): (k, l2norm(v)), word_vecs.iteritems()))

def print_sorted(pair_dict):
	sorted_lengths = sorted(word_lengths.items(), key = operator.itemgetter(1))
	print("\n".join(map(str, sorted_lengths[0:20])))
        print("\n")
	print("\n".join(map(str, sorted_lengths[-20:-1])))

print_sorted(word_lengths)

