import numpy as np
import scipy.stats as sps
import math
import itertools as it
from functools import wraps

def oneProportionZTtest(a,b):
	"""
	computes the p value of the indicator I(a<b) assuming enough samples to approximate normal.
	specifically, returns the P(a=b)
	"""
	p=np.sum(a<b)
	z=(p-0.5)*np.sqrt(len(a))/0.5
	return sps.binom(len(a),0.5).ppf(z)

"""
we have 3 main tests:
injecting noise on each dataset for each metric (chiecking increasing noise causes a changed metric)
comparing if shapes are the same (hypothesis test: distributions not equal)
comparing if different embeddings compare well against each other (same as beore, but on embeddings)
comparing if shapes-embeddings are similar (same as before..except want to prove are same)

this breaks down into 2 basic tasks for significance testing:
checking monotonicity
Hypothesis testing values

Approaches:
straight hypothesis testing
permutation testing.
"""

def permuteTest(treatmentLevel, expirimentalResult, ntest, functionToCompare):
	"""
	resources on permutation testing:
	http://thomasleeper.com/Rcourse/Tutorials/permutationtests.html
	http://ordination.okstate.edu/permute.htm
	idea: do an empirical hypothesis test under the assumption that the treatmentLevel is independent of the expirimental result.
	test this hypothesis by finding random permutations of the treatment-experimental pairs and then calculate a statistic with functionToCompare.
	Accumulate the different values of functionToCompare to provide a sample of what happens
	:param treatmentLevel:
	:param expirimentalResult:
	:param ntest:
	:param functionToCompare:
	:return:
	"""
	assert len(treatmentLevel.shape)==1 and len(expirimentalResult.shape)==1
	assert len(treatmentLevel)==len(expirimentalResult)
	if math.factorial(len(treatmentLevel))<ntest: #full factorial
		toTest=list(it.permutations(range(len(treatmentLevel))))
	else:
		toTest=generateShuffles(len(treatmentLevel),ntest)
	testRes=list(map(lambda permute: functionToCompare(treatmentLevel,expirimentalResult[permute]),toTest)) # perfrom permutation on expirimental result
	return np.array(testRes)

def generateShuffles(nelem, nshuffle):
	"""
	generates permutations without replacement (into the space of permutations) by rejection sampling.
	:param nelem:
	:param nshuffle:
	:return:
	"""
	accum=set()
	while len(accum)<nshuffle:
		thisShuffle=tuple(np.random.permutation(nelem).tolist())
		accum.add(thisShuffle)
	return list(accum)

def permutePtest(treatmentLevel, expirimentalResult, ntest, functionToCompare):
	permuteRes=permuteTest(treatmentLevel,expirimentalResult,ntest,functionToCompare)
	thisRes=functionToCompare(treatmentLevel,expirimentalResult)
	return np.mean(thisRes<permuteRes)

def spearmanPtest(treatmentLevel, expirimentalResult):
	assert len(treatmentLevel.shape)==1 and len(expirimentalResult.shape)==1
	assert len(treatmentLevel)==len(expirimentalResult)
	return sps.spearmanr(treatmentLevel,expirimentalResult)[1]

def resampleTest(treatmentLevel, expirimentalResult, ntest, functionToCompare):
	"""
	runs resampling to approximate the result of the spearman rank coefficent
	inpired by: https://www.mathworks.com/help/stats/resampling-statistics.html
	idea: find the spearman coefficient
	:return:
	"""
	assert len(treatmentLevel.shape)==1 and len(expirimentalResult.shape)==1
	assert len(treatmentLevel)==len(expirimentalResult)
	rankCoeffRsmplAccum=[]
	for test in range(ntest):
		randomSel=np.random.randint(0,len(treatmentLevel)-1,shape=(len(treatmentLevel),))
		rankCoeffRsmpl=functionToCompare(treatmentLevel[randomSel],expirimentalResult[randomSel])
		rankCoeffRsmplAccum.append(rankCoeffRsmpl)
	resampledArray=np.array(rankCoeffRsmplAccum)
	return resampledArray

"""
the following are for chekcing monotonicity
"""
#see sps.spearmanr. Note that it actually returns two values, the spearman's rho (correlation of ranks) AND the p value of the obseved rho under some kind of test (likely a permutation test)

"""
the following are intended to test if the values of different groups are the same
"""
def welchTtest(pop1,pop2):
	"""
	This is the proper test assuming that the mean is a good measure of centrality in noise.
	:param pop1:
	:param pop2:
	:return:
	"""
	return sps.ttest_ind(pop1, pop2, equal_var=False)

def __toTreatAndExpr(pop1, pop2):
	treat=np.concatenate((np.zeros(len(pop1)), np.ones(len(pop2))))
	expr=np.concatenate((pop1,pop2))
	return treat, expr
def __groupIntoPops(zeroOneGroups,vals):
	sortIndxs=np.argsort(zeroOneGroups)
	sortZeroOne=zeroOneGroups[sortIndxs]
	breakpt=np.where(np.diff(sortZeroOne)>0)
	sortVals=vals[sortIndxs]
	assert len(breakpt)==1
	pop1=sortVals[:(breakpt-1)]
	pop2=sortVals[breakpt:]
	assert(len(pop1) > 0 and len(pop2) >0 )
	return pop1, pop2

def __wrapFunction(f):
	@wraps(f)
	def wrappedF(zeroOneGroups, values):
		pop1, pop2=__groupIntoPops(zeroOneGroups, values)
		return f(pop1,pop2)
	return wrappedF

def permuteTestPop(pop1, pop2, ntest, functionToCompare):
	# is this correct? need to wrap functionToCompare to change argument types.
	# more to the point, the permutation testing is going to consider groupings of uneven size
	t,e=__toTreatAndExpr(pop1,pop2)
	return permuteTest(t,e,ntest,functionToCompare)
def permutePtestPop(pop1, pop2, ntest, functionToCompare):
	permuteRes=permuteTestPop(pop1,pop2,ntest,functionToCompare)
	thisRes=functionToCompare(pop1,pop2)
	return np.mean(thisRes<permuteRes)

def resampleTestPop(pop1, pop2, ntest, functionToCompare):
	t,e=__toTreatAndExpr(pop1,pop2)
	return resampleTest(t,e,ntest,functionToCompare)

