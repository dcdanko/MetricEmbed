import sys, math, numpy
from collections import Counter
from scipy.sparse import lil_matrix
import scipy.sparse.linalg
import fractal_dim as fd

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
vocab = [term for term in term_counter.keys() if (term_doc_counter[term] >= 5 and term_doc_counter[term] < 500)]
vocab_ids = {}

## Create a map from stings to indices into a vector
for term_id, term in enumerate(vocab):
    vocab_ids[term] = term_id
    
print "{} {}".format(num_docs, len(vocab))

## lil_matrix is a list of lists (NOT a cute adorable matrix). It's a good format for constructing a sparse matrix, but not great for computation.
doc_term_matrix = lil_matrix((num_docs, len(vocab)))
for doc_id in range(num_docs):
    doc_counter = docs[doc_id]
    
    ## Create a list of term IDs and a list of word-count values for this document
    terms = sorted([term for term in doc_counter if term in vocab_ids], key = lambda term: vocab_ids[term])
    term_ids = map(lambda term: vocab_ids[term], terms)
    values = map(lambda term: doc_counter[term], terms)

    ## Normalize the documents
    norm = numpy.sqrt(numpy.sum(map(lambda value : value ** 2, values)))
    if norm != 0:
        values = values / norm
    
    ## Create a sparse row, recording only non-zeros
    doc_term_matrix[doc_id,term_ids] = values

## Convert the lil_matrix to a more optimized format
doc_term_matrix = doc_term_matrix.tocsr()

print "running SVD"

## Calculate a truncated SVD
doc_matrix, singular_values, word_matrix = scipy.sparse.linalg.svds(doc_term_matrix, 50)
## Transpose the word_matrix so that columns correspond to latent dimensions in both matrices 
word_matrix = word_matrix.T

fd.fractalDimension(word_matrix.T,0.,.000125,400,'LSA')
fd.plotEmbeddings(['Embeddings/w2v1','Embeddings/w2v2','Embeddings/w2v3','Embeddings/w2v4','Embeddings/w2v5'])

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
    row_norms = numpy.sqrt(numpy.sum(doc_term_matrix ** 2), axis = 1)
    return numpy.delete(doc_term_matrix, (row_norms == 0.0).nonzero()[0], axis=0)
    
def l2_norm(matrix):
    row_norms = numpy.sqrt(numpy.sum(matrix ** 2, axis = 1))
    return matrix / row_norms[:, numpy.newaxis]

# Function for B5 - finds the closest docs to the input term
def find_closest_docs_to_term(term):
    source_id = vocab_ids[term]
    source_vector = word_matrix[source_id,:]
    sums = doc_matrix.dot(source_vector)
    print_sorted(sums, doc_snippets)

