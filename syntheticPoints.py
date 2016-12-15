import numpy as np
import itertools as it
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D #it's tempting. don't delete.

def hypercubeVerticies(dimSize):
    return np.array(list(it.product((0,1),repeat=dimSize)))

def randomHypercube(nsmpl, dimSize):
    return np.random.rand(nsmpl,dimSize)

def randomSphere(nsmpl, dimSize):
    variates= np.random.normal(size=(nsmpl, dimSize))
    variates/=np.linalg.norm(variates, axis=1)[:,np.newaxis]
    return variates

def randomInteriorSphere(nsmpl, dimSize):
    return (np.random.rand(nsmpl)[:,np.newaxis])**(1/dimSize)*randomSphere(nsmpl,dimSize)

def randomUnitSimplex(nsmpl, dimSize):
    preimage=np.hstack((np.zeros((nsmpl,1)), np.random.rand(nsmpl, dimSize-1), np.ones((nsmpl,1))))
    return np.diff(np.sort(preimage, axis=1), axis=1)

def unitSimplexVerticies(dimSize):
    for toBeOne in range(dimSize):
        ret=np.zeros(dimSize)
        ret[toBeOne]=1
        yield ret

def unitLatLonSphere(nlatLonTuple):
    """
    generates points on the n-sphere.
    :param nlatLonTuple: a tuple of (nlat, nlon) where (nlat, nlon) should be extended for however many dimensions is in the space -1. Gives the numebr of latitude and longitude. Sphere will have np.prod(nLatLonTuple) number of points.
    :return:
    """
    ndim=len(nlatLonTuple)+1
    sampleArrs=tuple(map(lambda n : np.linspace(0,2*np.pi,n+1,endpoint=False)[1:], nlatLonTuple))
    angles=np.array(list(it.product(*sampleArrs)))
    cumProd=np.ones(angles.shape[0])
    coordinates=np.zeros((np.product(nlatLonTuple),ndim))
    for dim in range(ndim-1):
        coordinates[:,dim]=np.cos(angles[:,dim])*cumProd
        cumProd*=np.sin(angles[:,dim])
    coordinates[:,ndim-1]=cumProd
    return coordinates

def randomCylinder(radii, heightOverRad, nsmpl):
    heights=np.random.rand(nsmpl)*heightOverRad
    circularPos=randomInteriorSphere(nsmpl, len(radii))
    return np.hstack((circularPos, heights[:,np.newaxis]))

def iterRandoms(nsmpl, dimSize):
    yield randomSphere(nsmpl, dimSize)
    yield randomHypercube(nsmpl, dimSize)
    yield randomUnitSimplex(nsmpl, dimSize)
    yield randomCylinder([1]*(dimSize-1), 10, nsmpl)

def iterRandomsLabels():
    yield 'random unit sphere'
    yield 'random unit hypercube'
    yield 'random unit simplex'
    yield 'random unit circular cylinder, height 10'

def draw3dSurface(points):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot(points[:,0], points[:,1], points[:,2], '.')
    ax.legend()
    plt.show()

if __name__=="__main__":
    print(list(hypercubeVerticies(5)))
    draw3dSurface(randomHypercube(1000,3))
    draw3dSurface(randomSphere(1000,3))
    draw3dSurface(randomCylinder((1,1),10,1000))
    draw3dSurface(randomUnitSimplex(1000,3))
    draw3dSurface(hypercubeVerticies(3))
    draw3dSurface(np.array(list(unitSimplexVerticies(3))))
    draw3dSurface(unitLatLonSphere((20,20)))
