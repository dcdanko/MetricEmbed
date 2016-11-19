import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #it's tempting. don't delete.

def hypercubeVerticies(dimSize):
    return it.product((0,1),repeat=dimSize)

def randomHypercube(nsmpl, dimSize):
    return np.random.rand(nsmpl,dimSize)

def randomSphere(nsmpl, dimSize):
    variates= np.random.normal(size=(nsmpl, dimSize))
    variates/=np.linalg.norm(variates, axis=1)[:,np.newaxis]
    return variates

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
