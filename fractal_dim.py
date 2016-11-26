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
    parser.add_argument('--sep',dest='sep',type=str,default=' ',help='Seperator for embedding tab;e')
    parser.add_argument('-s','--samples',dest='samples',type=int,default=1000,help='Number of points to use as centers')
    parser.add_argument('-r','--radius',dest='radius',type=float,default=[1.0],nargs='+',help='Radii to count points in')
    parser.add_argument('-e','--embedding',dest='embeddingf',type=str, help='File containing the word embedding')

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
import simpleMeasures as sm
import math

def findPointClouds(embeddings,nSamples,radii,metric):
    samples = embeddings.sample(n=nSamples,axis=1)
    distMatrix = scipy.spatial.distance.cdist(samples.T,embeddings.T,metric=metric)

    nPoints = {}
    for radius in radii:
        nPoints[radius] = np.less_equal(distMatrix,radius).sum(axis=1).mean()

    return nPoints

def findPointCloud(embedding,radius):
    """
    Computes the size of the point cloud of radius radius about the origin.
    """
    l2norms = np.sqrt(np.square(embedding).sum(axis=0))
    return sum(map(lambda x : 1 if x < radius else 0,l2norms))

def getMultiplicativeGrowthRate(cloudSizes):
    """
    Computes the average multiplicative growth rate (over indices)
    """
    sortedCloudSizes = filter(lambda x : x > 0.5, cloudSizes)
    ratios = np.divide(sortedCloudSizes[1:],sortedCloudSizes[:-1])
    return sum(ratios) / (len(sortedCloudSizes) - 1)

def fractalDimension(embedding,initRad,radFactor,radCount):
    """
    Computes the fractal dimension w.r.t. the origin of the given embedding.
    Scanns radii initRad*radFactor^0 -> initRad*radFactor^(radCount-1)
    """
    avgDist = sm.averagePairwiseDistance(embedding)
    cloudSizes = np.zeros(radCount)
    radius = initRad * avgDist
    for i in range(0,radCount):
        cloudSizes[i] = findPointCloud(embedding,radius)
        radius = radius * radFactor
    growthRate = getMultiplicativeGrowthRate(cloudSizes)
    return math.log(growthRate,radFactor)

def main():
    args = buildArgs()
    #embeddings = pd.DataFrame.from_csv(args.embeddingf,sep=args.sep,header=None)
    embeddings = pd.DataFrame.from_dict(parse.parse(args.embeddingf))
    dim = fractalDimension(embeddings,0.01,1.2,10)
    print('{}: {}'.format('Fractal Dimension',dim))
    #cloudSizes = findPointClouds(embeddings,args.samples,args.radius,'euclidean')
    #for r,s in cloudSizes.items():
    #    print('{},{}'.format(r,s))
    
################################################################################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
