import sys, math, numpy
from collections import Counter
import pandas as pd

num_topics = 50
num_iterations = 50

## This Counter will record total term frequencies -- the number of 
##  tokens of each type in the corpus
term_counter = Counter()

## This Counter will store the document frequencies -- the number of 
##  documents that contain at least one instance of each term
term_doc_counter = Counter()

## This will be the total number of documents
num_docs = 0
doc_snippets = []
docs = []

print "reading"

## Count words in documents
for line in open(sys.argv[1]):
    line = line.replace(".", " ").replace(",", " ")
    num_docs += 1
    doc_counter = Counter(line.split())
    
    for term in doc_counter:
        term_counter[term] += doc_counter[term]
        term_doc_counter[term] += 1

    # These should be equivalent, but were MUCH slower
    #term_counter += doc_counter
    #term_doc_counter += Counter(doc_counter.keys())
    
    doc_snippets.append(line[:200])
    docs.append(doc_counter)
    
## Remove infrequent terms
vocab = [term for term in term_counter.keys() if term_doc_counter[term] >= 5 and term_doc_counter[term] < num_docs * 0.1 and len(term) > 2]
vocab_ids = {}

## Create a map from stings to indices into a vector
for term_id, term in enumerate(vocab):
    vocab_ids[term] = term_id
    
num_terms = len(vocab)

print "{} {}".format(num_docs, num_terms)

term_topics = numpy.ones((num_terms, num_topics)) * 1.0 / num_terms
topic_docs = numpy.ones((num_topics, num_docs)) * 1.0 / num_topics

for topic in range(num_topics):
	term_topics[:,topic] = numpy.random.dirichlet(numpy.ones(num_terms))

for iteration in range(num_iterations):
	objective = 0.0
	
	new_term_topics = numpy.zeros((num_terms, num_topics))
	new_topic_docs = numpy.zeros((num_topics, num_docs))
	
	for doc_id in range(num_docs):
		doc_counter = docs[doc_id]
		
		for term in doc_counter.keys():
			if term in vocab_ids:
			
				term_id = vocab_ids[term]
			
				### ADD CODE HERE
				### Compute q_dw for term/doc (vector)
				### Calculate update of theta, phi
				### Update Objective function

				q = term_topics[term_id,:] * topic_docs[:,doc_id]
				q_sum = numpy.sum(q)
				q /= q_sum

				new_term_topics[term_id,:] += doc_counter[term] * q
				new_topic_docs[:, doc_id] += doc_counter[term] * q

				objective += doc_counter[term] * math.log(q_sum)

		## Check for empty documents
		if numpy.sum(new_topic_docs[:,doc_id]) == 0.0:
			new_topic_docs[:,doc_id] = 1.0 / num_topics
			
	print "iteration {}, {}".format(iteration, objective)
			
	### ADD CODE HERE
	### Normalize

	new_term_topics = new_term_topics / new_term_topics.sum(axis = 0)
	new_topic_docs = new_topic_docs / new_topic_docs.sum(axis = 0)

	term_topics = new_term_topics
	topic_docs = new_topic_docs
	
word_vecs = {}
for term,term_id in vocab_ids.iteritems():
    word_vecs[term] = term_topics[term_id,:]
df = pd.DataFrame.from_dict(word_vecs,orient='index')
df.to_csv(sys.argv[2], sep=' ')

def print_sorted(x, labels):
    sorted_labels = sorted(enumerate(x), key=lambda w: -w[1])
    for index, score in sorted_labels[0:20]:
        label = labels[index]
        print index, score, label

def find_closest_terms(term):
    source_id = vocab_ids[term]
    
    source_vector = word_matrix[source_id,:]
    sums = word_matrix.dot(source_vector)
    print_sorted(sums, vocab)
    
def find_closest_docs(source_id):
    source_vector = doc_matrix[source_id,:]
    sums = doc_matrix.dot(source_vector)
    print_sorted(sums, doc_snippets)


## Functions that return a modified copy of a matrix
def remove_empty_rows():
    row_norms = numpy.sqrt(numpy.sum(doc_term_matrix ** 2, axis = 1))
    return numpy.delete(doc_term_matrix, (row_norms == 0.0).nonzero()[0], axis=0)
    
def l2_norm(matrix):
    row_norms = numpy.sqrt(numpy.sum(matrix ** 2, axis = 1))
    return matrix / row_norms[:, numpy.newaxis]

