
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
    parser = argparse.ArgumentParser(description='Calculate metric-entropy based distance between two word embeddings')
    parser.add_argument('--num-trials',dest='ntrials',type=int,default=5,help='Number of trials to run when calculating metric entropy.')
    parser.add_argument('--min-radius', dest='min_radius',type=float, default=0.01, help='Minimum radius to use as a proportion of ave pairwise distance')
    parser.add_argument('--max-radius', dest='max_radius',type=float, default=0.5, help='Maximum radius to use as a proportion of ave pairwise distance')
    parser.add_argument('--radius-step', dest='step',type=float, default=0.05, help='Step between min and max radii')
    parser.add_argument('--pair-metric', dest='pair_metric',type=str, default='euclidean', help='Pairwise distance metric which will be used')
    parser.add_argument('--vec-metric', dest='vec_metric',type=str, default='jsd', help='Vector distance metric which will be used')
    parser.add_argument('first_embedding',type=str,help='File containing the first embedding')
    parser.add_argument('second_embedding',type=str,help='File containing the second embedding')
    
    parser.add_argument('--test',action='store_true',help="Run the program's built in tests. All other arguments will be ignored.")
    args = parser.parse_args()
    return args

################################################################################
#
# Main Code
#
################################################################################

import sys
import parse

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cdist, pdist
from metric_entropy import estimateMetricEntropy
import pandas as pd

def main():
    args = buildArgs()
    emb1 = args.first_embedding
    emb2 = args.second_embedding
    emb1 = pd.DataFrame.from_dict(parse.parse(emb1))
    emb2 = pd.DataFrame.from_dict(parse.parse(emb2))
    D = compareMetricEntropies(emb1,
                               emb2,
                               minR=args.min_radius,
                               maxR=args.max_radius,
                               step=args.step,
                               pairMetric=args.pair_metric,
                               vecMetric=args.vec_metric,
                               ntrials=args.ntrials)
    print("\n{}".format(D))
                               
def getMetricEntropyVec(embedding,minR,maxR,step,pairMetric,ntrials,normRange=True):
    avePairDist = pdist(embedding,metric=pairMetric).mean()
    minR = minR * avePairDist
    maxR = maxR * avePairDist
    step = step * avePairDist
    radii = np.arange(minR,maxR+step,step)
    metricEntropies = []
    for r in radii:
        metricEntropy = estimateMetricEntropy(embedding,r,metric=pairMetric,ntrials=ntrials)
        metricEntropies.append( metricEntropy)

    metricEntropies = np.array( metricEntropies)
    if normRange:
        norm = np.linalg.norm( metricEntropies)
        metricEntropies = metricEntropies / norm
    return metricEntropies


def compareVecs(mvec1,mvec2,vecMetric):
    vecMetric = vecMetric.lower()
    if vecMetric in ['euclidean','l2']:
        D = cdist(np.array(mvec1,ndmin=2),np.array(mvec2,ndmin=2),'euclidean')[0,0]
    elif vecMetric in ['mean','mean-diff','meandiff']:
        D = abs(mvec1.mean() - mvec2.mean())
    elif vecMetric in ['manhattan','l1']:
        D = cdist(np.array(mvec1,ndmin=2),np.array(mvec2,ndmin=2),'cityblock')[0,0]
    elif vecMetric in ['jsd']:
        M = 0.5 * (mvec1 + mvec2)
        D = 0.5 * (entropy(mvec1,M) + entropy(mvec2,M))
    else:
        D = cdist(np.array(mvec1,ndmin=2),np.array(mvec2,ndmin=2),'vecMetric')[0,0]
    return D

def compareMetricEntropies(emb1,emb2,
                           minR=0.5,
                           maxR=2,
                           step=0.1,
                           pairMetric='euclidean',
                           vecMetric='jsd',
                           ntrials=5):
    mvec1 = getMetricEntropyVec(emb1,minR,maxR,step,pairMetric,ntrials)
    mvec2 = getMetricEntropyVec(emb2,minR,maxR,step,pairMetric,ntrials)


    D = compareVecs(mvec1,mvec2,vecMetric)
    return D
    





    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)