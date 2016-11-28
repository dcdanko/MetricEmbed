import numpy as np
import sys
import parse
from collections import namedtuple

def allPairwiseDistSqrd(embed1, embed2):
    """
    computes a numpy array with all pairwise distances between the points in embeddings in euclidean space
    observe if use the same embedding as both arguments, will get all pairwise distances within the space
    technically returns the distance squared to eliminate roundoff errors. if need actual distance, take square root.
    :param embed1: An embedding of points given as a linear combination of some kind of basis. each row is an element (point) and each column is a component. For efficiency, C-ordering is preferred
    :param embed2: Assumes common basis with embed1
    :return:
    """
    distances = (np.sum(embed1**2, axis=1)[:,np.newaxis])+(np.sum(embed2**2, axis=1)[:,np.newaxis]).T-2*np.dot(embed1,embed2.T)
    distances[np.logical_and(distances < 0, distances > -1e-8)]=0
    return distances

def averagePairwiseDistance(embed):
    """
    Computes the average pairwise distance of the input embedding.
    """
    return np.mean(np.sqrt(allPairwiseDistSqrd(embed, embed)))

def pairwiseDistanceChange(embed1, embed2):
    """

    :return:
    """
    # assume the data is in the same order in each embedding (i.e. element 0 in embed1 corresponds to the same element 0 in embed2)
    return np.sqrt(allPairwiseDistSqrd(embed1, embed1))-np.sqrt(allPairwiseDistSqrd(embed2, embed2))

def minPairwiseDistChange(embed1,embed2):
    """
    solves the linear optimization problem defined by:
    min sum ((alpha * sqrt(allPairwiseDistSqrt(embed1,embed1)) - (1-alpha) * sqrt(allPairwiseDistSqrd(embed2,embed2))))^2
    taking the matricies are vectors

    in effect, we are allowed to globally rescale the pairwise distances to find a better match

    this differs from pairwiseDistanceChange in that pairwiseDistanceChange retains the original distances which is closer to what would be reported numerically if were copmaring the embeddings.
    This rescales things in an attempt to normalize the fact we look at the rankings when, for example, we would take the top 5 for an analogy task.
    :param embed1:
    :param embed2:
    :return:
    """
    avgVect=(embed1.flatten()+embed2.flatten())/2
    flat2=embed2.flatten()
    optScale=np.dot(avgVect,flat2)/np.dot(avgVect,avgVect)
    return optScale*avgVect-flat2

sumStats=namedtuple('sumStats','mean stddev median min max')
def getSumStats(elements):
    return sumStats(np.mean(elements), np.std(elements), np.median(elements), np.min(elements), np.max(elements))

unitize=lambda sampleMat: sampleMat/(np.linalg.norm(sampleMat,axis=1)[:,np.newaxis])

if __name__=="__main__":
    dict1 = parse.parse(sys.argv[1])
    dict2 = parse.parse(sys.argv[2])
    sorted_vocab = sorted(dict1)
    embed1 = np.array(map(lambda word: dict1[word], sorted_vocab))
    embed2 = np.array(map(lambda word: dict2[word], sorted_vocab))
    print('ordinary tests')
    #print(getSumStats(np.abs(pairwiseDistanceChange(embed1, embed2))))
    print(getSumStats(pairwiseDistanceChange(embed1, embed2)**2))
    #print(getSumStats(np.abs(minPairwiseDistChange(embed1, embed2))))
    print(getSumStats(minPairwiseDistChange(embed1, embed2)**2))

    # for cosine
    normTestEmbed1=unitize(embed1)
    normTestEmbed2=unitize(embed2)
    print('cosine tests')
    #print(getSumStats(np.abs(pairwiseDistanceChange(normTestEmbed1, normTestEmbed2))))
    print(getSumStats(pairwiseDistanceChange(normTestEmbed1, normTestEmbed2)**2))
    #print(getSumStats(np.abs(minPairwiseDistChange(normTestEmbed1, normTestEmbed2))))
    print(getSumStats(minPairwiseDistChange(normTestEmbed1, normTestEmbed2)**2))
