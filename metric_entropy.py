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
# Data Structures
#
################################################################################

################################################################################
#
# Data Structures
#
################################################################################

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

from matplotlib.pyplot import hist
from matplotlib.pyplot import scatter

class Sphere:
    def __init__(self,centerWord,centerVec):
        self.centerWord = centerWord
        self.centerVec = centerVec
        self.points = [(centerWord,centerVec)]

    def add(self,centerWord,centerVec):
        self.points.append( (centerWord,centerVec))
        

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
        
def printSummaryOfCoveringSpheres(coveringSpheres,radius,drawHist=False):
#    print('Distance function: {}'.format(distFunc))
    print('Radius: {}'.format(radius))
    print('Total spheres: {}'.format(len(coveringSpheres)))
    print('Singletons: {}'.format(len([sphere for sphere in coveringSpheres if len(sphere.points) == 1])))
    sphereSizes = [len(sphere.points) for sphere in coveringSpheres]
    print('Total points covered: {}'.format(sum( sphereSizes)))
    if drawHist:
        hist(sphereSizes,bincount=max(sphereSizes)/10,xlab=True)




def main():
    args = buildArgs()
    #embeddings = pd.DataFrame.from_csv(args.embeddingf,sep=args.sep,header=None)
    embeddings = pd.DataFrame.from_dict(parse.parse(args.embeddingf))
    nSpheres = []
    for radius in args.radius:
        coveringSpheres = findCoveringSpheres(embeddings.T,scipy.spatial.distance.euclidean,radius)
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
