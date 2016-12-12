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
import embed_parse


def findPointClouds(embeddings,nSamples,radii,metric):
    samples = embeddings.sample(n=nSamples,axis=1)
    distMatrix = scipy.spatial.distance.cdist(samples.T,embeddings.T,metric=metric)

    nPoints = {}
    for radius in radii:
        nPoints[radius] = np.less_equal(distMatrix,radius).sum(axis=1).mean()

    return nPoints





def main():
    args = buildArgs()
    #embeddings = pd.DataFrame.from_csv(args.embeddingf,sep=args.sep,header=None)
    embeddings = pd.DataFrame.from_dict(embed_parse.parse(args.embeddingf))
    cloudSizes = findPointClouds(embeddings,args.samples,args.radius,'euclidean')
    for r,s in cloudSizes.items():
        print('{},{}'.format(r,s))
    
################################################################################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
