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
import numpy.matlib
import scipy.spatial.distance
import subprocess as sp
import embed_parse as ep
import simpleMeasures as sm
import math
import matplotlib.pyplot as plt
import syntheticPoints as synp

def findPointClouds(embeddings,nSamples,radii,metric):
    samples = embeddings.sample(n=nSamples,axis=1)
    distMatrix = scipy.spatial.distance.cdist(samples.T,embeddings.T,metric=metric)

    nPoints = {}
    for radius in radii:
        nPoints[radius] = np.less_equal(distMatrix,radius).sum(axis=1).mean()

    return nPoints

def findPointCloud(embedding,radius,center):
    """
    Computes the size of the point cloud of radius radius about the origin.
    """
    repCenter = np.matlib.repmat(center,embedding.shape[1],1).T
    l2norms = np.sqrt(np.square(embedding-repCenter).sum(axis=0))
    return sum(map(lambda x : 1 if x < radius else 0,l2norms))

def getMultiplicativeGrowthRate(cloudSizes):
    """
    Computes the average multiplicative growth rate (over indices)
    """
    #print(cloudSizes)
    #sortedCloudSizes = filter(lambda x : x > 0.5, cloudSizes)
    sortedRadii = filter(lambda x : cloudSizes[x] > 1.5, sorted(cloudSizes))
    sortedCloudSizes = filter(lambda x : x > 1.5, map(lambda x : float(cloudSizes[x]), sortedRadii))
    #sortedRadii = sorted(cloudSizes)
    #sortedCloudSizes = map(lambda x : float(cloudSizes[x]), sortedRadii)
    #print(sortedCloudSizes)
    countRatios = np.divide(sortedCloudSizes[1:],sortedCloudSizes[:-1])
    radiusRatios = np.divide(sortedRadii[1:],sortedRadii[:-1])
    logs = np.divide(np.log(countRatios),np.log(radiusRatios))
    
    # PLOTTING HERE
    #plt.plot(logs)
    #plt.plot(sortedCloudSizes[1:],logs)
    #plt.show()
    # END PLOTTING

    #return sum(countRatios) / (len(sortedCloudSizes) - 1)
    #return sum(logs) / (len(sortedRadii) - 1)
    return sortedCloudSizes[1:],logs

def fractalDimension(embedding,initRad,radFactor,radCount,eName):
    """
    Computes the fractal dimension w.r.t. the origin of the given embedding.
    Scanns radii initRad*radFactor^0 -> initRad*radFactor^(radCount-1)
    """
    avgDist = sm.averagePairwiseDistance(embedding)
    embedCenter = np.sum(embedding,axis=1)/(embedding.shape[1])
    #samples = embedding.sample(10,axis=1)
    center = (.002*avgDist*np.random.randn(embedding.shape[0])) + embedCenter
    cloudSizes = {}
    radius = initRad * avgDist
    for i in range(0,radCount):
        #cloudSizes[i] = findPointCloud(embedding,radius,center)
        cloudSizes[radius] = findPointCloud(embedding,radius,center)
        #radius = radius * radFactor
        radius = radius + (radFactor * avgDist)
    sizes,logs = getMultiplicativeGrowthRate(cloudSizes)
    plt.plot(sizes,logs,label=eName)
    #return math.log(growthRate,radFactor)
    #return growthRate

def globalFractalDimension(embedding,numSamples,initRad,radFactor,radCount,eName):
    embeddingSize = embedding.shape[1]
    avgDist = sm.averagePairwiseDistance(embedding)
    centers = embedding.sample(numSamples,axis=1)
    distMatrix = scipy.spatial.distance.cdist(centers.T,embedding.T,metric='euclidean')
    mins = np.amin(distMatrix, axis=0)
    cloudSizes = {}
    radius = initRad * avgDist
    for i in range(0,radCount):
        cloudSizes[radius] = np.sum(map(lambda x : 1 if x <= radius else 0, mins))
        radius = radius + (radFactor * avgDist)
    sizes,logs = getMultiplicativeGrowthRate(cloudSizes)
    plt.plot(sizes,logs,label=eName)

def plotEmbeddings(files):
    for filename in files:
        embedding=pd.DataFrame.from_dict(ep.parse(filename))
        #fractalDimension(embedding,0.,.00025,200,filename)
        globalFractalDimension(embedding,10,0.,.00025,200,filename)
    plt.ylabel('fractal dimension')
    plt.xlabel('points seen')
    plt.legend()
    plt.show()

def main():
    args = buildArgs()
    #embeddings = pd.DataFrame.from_csv(args.embeddingf,sep=args.sep,header=None)
    #embeddings = pd.DataFrame.from_dict(ep.parse(args.embeddingf))
    #dim = fractalDimension(embeddings,0.,.00025,200)
    #fractalDimension(synp.randomHypercube(200000,50).T,0.,.0005,400)
    fractalDimension(synp.randomHypercube(7000,50).T,0.,.0005,400,'cube1')
    globalFractalDimension(pd.DataFrame.from_dict(synp.randomHypercube(7000,50).T),10,0.,.0005,400,'cube2')
    globalFractalDimension(pd.DataFrame.from_dict(synp.randomSphere(7000,50).T),10,0.,.0005,400,'sphere2')
    fractalDimension(synp.randomSphere(7000,50).T,0.,.0005,400,'sphere1')
    #print('{}: {}'.format('Fractal Dimension',dim))
    plt.legend()
    plt.show()
    #cloudSizes = findPointClouds(embeddings,args.samples,args.radius,'euclidean')
    #for r,s in cloudSizes.items():
    #    print('{},{}'.format(r,s))
    
################################################################################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
