#!/usr/bin/env python

################################################################################
#
# Arguments and Utility Functions
#
################################################################################

import gzip
import argparse

class cols:
    HEADER = '[95m'
    OKBLUE = '[94m'
    OKGREEN = '[92m'
    WARNING = '[93m'
    FAIL = '[91m'
    ENDC = '[0m'
    BOLD = '[1m'
    UNDERLINE = '[4m'


def gopen(fname, mode='r'):
    if fname[:-3] == '.gz':
        return gzip.open(fname,mode)
    return open(fname,mode)

def buildArgs():
    parser = argparse.ArgumentParser(description='A default script')
    parser.add_argument('-s','--seperator',dest='sep',type=str,default=' ',help='Character to seperate columns in embedding file')
    parser.add_argument('-r','--radius',dest='radius',type=float,default=[1.0],nargs='+',help='Radius of covering spheres')
    parser.add_argument('-e','--embedding',dest='embeddingf',type=str, help='File containing the word embedding')

    parser.add_argument('--hist',dest='hist',action='store_true',help='Draw a histogram of number of points in spheres')

    args = parser.parse_args()
    return args


################################################################################
#
# Main Code
#
################################################################################

import sys
import pandas as pd
import numpy as np
import scipy.spatial.distance
import subprocess as sp
import parse

class Sphere:
    def __init__(self,centerWord,centerVec):
        self.centerWord = centerWord
        self.centerVec = centerVec
        self.points = [(centerWord,centerVec)]

    def add(self,centerWord,centerVec):
        if centerWord!= self.centerWord:
            self.points.append( (centerWord,centerVec))
        else:
            print('warning: attempted to re-add centerword. skipping')

    def extend(self, centerWordIt, centerVecIt):
        for centerWord, centerVec in zip(centerWordIt, centerVecIt):
            self.add(centerWord,centerVec)

def findCoveringSpheres(embeddings,dist,maxRadius):
    coveringSpheres = []
    shuffledEmbed = embeddings.reindex( np.random.permutation( embeddings.index))

    for i, (word, wordVec) in enumerate(shuffledEmbed.iterrows()):
        addedToSphere = False
        for sphere in coveringSpheres:
            if dist(sphere.centerVec,wordVec) <= maxRadius:
                sphere.add(word,wordVec)
                addedToSphere = True
                break
        if not addedToSphere:
            newSphere = Sphere(word,wordVec)
            coveringSpheres.append(newSphere)
        if i % 100 == 0:
            if i % 1000 == 0:
                sys.stderr.write('\r')
            else:
                sys.stderr.write('.')
    sys.stderr.write('\n')
    return coveringSpheres

def findCoveringSpheres_numpy(embeddings, metricChoice, maxRadius):
    """ alternative implementation of findCoveringSpheres that using numpy in the inner loop

    should be faster

    :param: embeddings: a pandas dataframe with the embedding locations
    :param: metricChoice: a distance function choice (see "metric" optional arguments in scipy.spatial.distance.cdist)
    :param: maxRadius: radius for the spheres.
    :return returns the tuple of words used as the centers of sphere and a numpy array with the sphere centers.
    """
    sphereWords=[]
    sphereCenters=np.zeros(embeddings.shape) # over-allocate
    shuffledEmbed = embeddings.reindex( np.random.permutation( embeddings.index))

    iterator=enumerate(shuffledEmbed.iterrows())
    i, (word, wordVec)=next(iterator)
    sphereWords.append(word)
    sphereCenters[0,:]=wordVec
    for i, (word, wordVec) in iterator:
        if np.all(scipy.spatial.distance.cdist(wordVec[np.newaxis,:], sphereCenters[0:len(sphereWords),:], metric=metricChoice)>maxRadius): # calculate using builtins and numpy
            sphereWords.append(word)
            sphereCenters[len(sphereWords)-1,:]=wordVec
        if i % 100 == 0:
            if i % 1000 == 0:
                sys.stderr.write('\r')
            else:
                sys.stderr.write('.')
    sphereCenters=sphereCenters[0:len(sphereWords),:]
    
    return (sphereWords, sphereCenters)

def findCoveringSpheres_inSphere(embeddings_numpy, centerLocations, metricChoice, maxRadius):
    """ finds which words are in which sphere using numpy

    should be faster, even if computing more data (numpy is really fast if done properly)
    :param embeddings_numpy: a numpy array with the locations of the embedded words
    :param centerLocations: a numpy array of the locations of hte sphere centers
    :param: metricChoice: a distance function choice (see "metric" optional arguments in scipy.spatial.distance.cdist)
    :param: maxRadius: radius for the spheres.
    :return: a boolean matrix where an entry is 1 iff the word (1st index) is in the sphere (2nd index)
    """
    return scipy.spatial.distance.cdist(embeddings_numpy, centerLocations, metric=metricChoice) <= maxRadius

def findCoveringSpheres_fast(embeddings, metricChoice, maxRadius):
    (centWords, centLocs)=findCoveringSpheres_numpy(embeddings, metricChoice, maxRadius)
    unusedWords=embeddings[~ embeddings.index.isin(centWords)] # already have centerwords in the spheres. What about others? Gets dataframe of unused words
    inSphere=findCoveringSpheres_inSphere(unusedWords.as_matrix(), centLocs, metricChoice, maxRadius) # horrible brute force copmutation, but C is faster than Python and numpy is in C.

    # go through and append words which are in each sphere. This shouldn't be faster than just iterating through, but funny things happen in python, so let's see...
    returnable=[]
    for sphereIndx, (centWord, centLoc) in enumerate(zip(centWords, centLocs)):
        thisSphere=Sphere(centWord, centLoc)
        inThisSphere=inSphere[:,sphereIndx]
        thisSphere.extend(unusedWords.index[inThisSphere], unusedWords.loc[inThisSphere].values)
        returnable.append(thisSphere)
    return returnable

def printSummaryOfCoveringSpheres(coveringSpheres,radius,drawHist=False):
#    print('Distance function: {}'.format(distFunc))
    print('Radius: {}'.format(radius))
    print('Total spheres: {}'.format(len(coveringSpheres)))
    print('Singletons: {}'.format(len([sphere for sphere in coveringSpheres if len(sphere.points) == 1])))
    sphereSizes = [len(sphere.points) for sphere in coveringSpheres]
    print('Total points covered: {}'.format(sum( sphereSizes)))
    if drawHist:
        hist(sphereSizes,bincount=max(sphereSizes)/10,xlab=True)

def estimateMetricEntropy(embeddings,radius,metric='euclidean',ntrials=5):
    estimates = []
    for arb in range(ntrials):
        coveringSpheres = findCoveringSpheres_fast(embeddings.T,metric, radius)
        estimates.append(len(coveringSpheres))

    return min(estimates)



def main():
    args = buildArgs()
    #embeddings = pd.DataFrame.from_csv(args.embeddingf,sep=args.sep,header=None)
    embeddings = pd.DataFrame.from_dict(parse.parse(args.embeddingf))
    nSpheres = []
    for radius in args.radius:
        # coveringSpheres = findCoveringSpheres(embeddings.T,scipy.spatial.distance.euclidean,radius)
        coveringSpheres = findCoveringSpheres_fast(embeddings.T,'euclidean', radius)
        printSummaryOfCoveringSpheres(coveringSpheres,args.radius,drawHist=args.hist)
        nSpheres.append((radius,len(coveringSpheres)))

    coverf = 'covers.txt'
    with open(coverf,'w') as cf:
        for r,s in nSpheres:
            cf.write('{},{}\n'.format(r,s))
#    scatter(coverf)
    

################################################################################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
