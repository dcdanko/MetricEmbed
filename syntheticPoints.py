import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #it's tempting. don't delete.

def hypercubeVerticies(dimSize):
    yield np.array(it.product((0,1),repeat=dimSize))

def randomHypercube(nsmpl, dimSize):
    return np.random.rand(nsmpl,dimSize)

def randomSphere(nsmpl, dimSize):
    variates= np.random.normal(size=(nsmpl, dimSize))
    variates/=np.linalg.norm(variates, axis=1)[:,np.newaxis]
    return variates

def randomUnitSimplex(nsmpl, dimSize):
    preimage=np.hstack((np.zeros((nsmpl,1)), np.random.rand(nsmpl, dimSize-1), np.ones((nsmpl,1))))
    return np.diff(np.sort(preimage, axis=1), axis=1)

def unitSimplexVerticies(dimSize):
    for toBeOne in range(dimSize):
        ret=np.zeros(dimSize)
        ret[toBeOne]=1
        yield ret

def draw3dSurface(points):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot(points[:,0], points[:,1], points[:,2], '.')
    ax.legend()
    plt.show()

if __name__=="__main__":
    print(list(hypercubeVerticies(5)))
    draw3dSurface(randomHypercube(1000,3))
    plt.figure()
    draw3dSurface(randomSphere(1000,3))
