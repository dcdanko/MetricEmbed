import numpy as np
import sys
import embed_parse
from collections import namedtuple
import scipy.spatial.distance as spd
import scipy.stats as sps
from functools import wraps

disableDistribute=False # global setting to disable functions which have distributeNumpy decorating them from being broken and computed
def distributeNumpy(numArraySize=int(50e3), accumulator=None, initialVal=None, finalStack=True):
	"""
	decorator for taking what should be a numpy operation and breaking it into a set of sequential operations performed blocks of rows
	assumes the decorated function takes in a sequence of numpy arrays which are all of the same number of rows.
	kwargs are assumed constant arguments

	defaults are chosen to make this look as close as possible to the usual case of running computations on a bunch of arrays an getting a single array out as a result
	other settings make sense in certain situations (e.g. taking a sum)

	untested.
	to make practical use of this might require some currying or other tricks. don't forget it is possible to call this funciton and get the decorated function out for making new function variants
	TODO: write a version of this that does products of a pair of arrays
	:param numArraySize: number of rows to do in a single call to the wrapped function
	:param accumulator: accumulation function to apply. takes 2 arguments, the first is being accumulated into. If set to None, will return a list of all results
	:param initialVal: initial value for acumulator.
	:param finalStack: accumulate into overall numpy array by re-merging on the first axis. Only really useful if using default settings for other parameters
	:return: the accumulated values from reduce(accumlator, map(decoratedFunction, argsBrokenIntoNumArraySizeGroups), initial=initialVal). if finalStack==True, call numpy.concatenate to get a single numpy array
	"""
	assert accumulator is None and initialVal is not None
	if accumulator is None:
		initialVal=[]
		accumulator=list.append
		if finalStack:
			stack=True
	if not finalStack:
		stack=False
	else:
		stack=True
	def distributeNumpy_decorator(f):
		if not disableDistribute:
			@wraps(f)
			def distributedF(*args, **kwargs):
				numElem=args[0].shape[0]
				indxBreaks=list(range(0,numElem,numArraySize))
				prevIndx=0
				accumResult=initialVal
				for indx in indxBreaks:
					thisArgs=tuple(thisArg[prevIndx:indx,:] for thisArg in args)
					thisResult=f(*thisArgs, **kwargs)
					accumulator(accumResult,thisResult)
					prevIndx=indx
				thisArgs=tuple(thisArg[prevIndx:thisArg.shape[0],:] for thisArg in args)
				thisResult=f(*thisArgs, **kwargs)
				accumulator(accumResult,thisResult)
				if stack:
					np.concatenate(accumulator,axis=0)
				return accumResult
			return distributedF
		else:
			return f
	return distributeNumpy_decorator

def orderingStability(embed1, embed2):
	e1dists=allPairwiseDistSqrd(embed1, embed1)
	e2dists=allPairwiseDistSqrd(embed2, embed2)
	return sps.spearmanr(e1dists, e2dists, axis=0)

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
	distances[np.logical_and(distances < 0, distances > -1e-12)]=0
	return distances

def allPairwiseOrdering(embed1,embed2):
	"""
	returns an array where each row is a word in embed1 (in the same order) and each element of each row is an index corresopnding to a word in embed2. The rows are order from the closest word to the farthest
	:param embed1:
	:param embed2:
	:return:
	"""
	dist=allPairwiseOrdering(embed1, embed2)
	ordering=np.argsort(dist, axis=1)
	return ordering

def averagePairwiseDistance(embed):
	"""
	Computes the average pairwise distance of the input embedding.
	"""
	return np.mean(np.sqrt(allPairwiseDistSqrd(embed, embed)))

def averageSquaredPairwiseDistanceChange(embed1, embed2):
   return np.mean(pairwiseDistanceChange(embed1, embed2)**2)

def pairwiseDistanceChange(embed1, embed2):
	"""

	:return:
	"""
	# assume the data is in the same order in each embedding (i.e. element 0 in embed1 corresponds to the same element 0 in embed2)
	return np.sqrt(allPairwiseDistSqrd(embed1, embed1))-np.sqrt(allPairwiseDistSqrd(embed2, embed2))

def closePoints(embed):
	allDist=allPairwiseDistSqrd(embed, embed)
	closest=np.min(allDist,axis=1)
	return closest

def __closePointCost(embedIntoMatrix, embedCostMatrix):
	closestWord=np.argmin(embedIntoMatrix,axis=1)
	return np.mean(embedCostMatrix[np.arange(len(closestWord)), closestWord]-embedIntoMatrix[np.arange(len(closestWord)), closestWord])
def closePointsChange(embed1, embed2):
	"""
	computes the closest point to each point and then uses distance between the word-pair in the other embedding as a "cost"
	then subtrcts the distance for the word-pair in the original embedding and computes the "regret"
	then does it the other way around to get a "total cost"

	represents the distance from the original embedding
	negative numbers indicate that the word-pairs actually moved closer on the average
	:param embed1:
	:param embed2:
	:return:
	"""
	allDist1=np.sqrt(allPairwiseDistSqrd(embed1, embed1))
	allDist1[np.arange(allDist1.shape[0]), np.arange(allDist1.shape[1])]=1e10 # should be big enough.
	allDist2=np.sqrt(allPairwiseDistSqrd(embed2, embed2))
	allDist2[np.arange(allDist2.shape[0]), np.arange(allDist2.shape[1])]=1e10 # should be big enough.
	return (__closePointCost(allDist1, allDist2)+__closePointCost(allDist1, allDist2))/2

# the following method looks plausible and runs, but the results DO NOT make sens for a metric (no symmtetric, for starters). I think I messed up my math.
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
	return optScale

sumStats=namedtuple('sumStats','mean stddev median min max')
def getSumStats(elements):
	return sumStats(np.mean(elements), np.std(elements), np.median(elements), np.min(elements[np.logical_not(np.isclose(elements, 0))]), np.max(elements))
def sumStatsListToSumStatsArrs(sumStatsList):
	meanVect=np.zeros(len(sumStatsList))
	stdVect=np.zeros(len(sumStatsList))
	medianVect=np.zeros(len(sumStatsList))
	minVect=np.zeros(len(sumStatsList))
	maxVect=np.zeros(len(sumStatsList))
	return (meanVect, stdVect, medianVect, minVect, maxVect)

unitize=lambda sampleMat: sampleMat/(np.linalg.norm(sampleMat,axis=1)[:,np.newaxis])

def runTests(embed1, embed2):
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

if __name__=="__main__":
	embeddings=embed_parse.importTwo(sys.argv[1], sys.argv[2])
	runTests(embeddings[0].evect, embeddings)
