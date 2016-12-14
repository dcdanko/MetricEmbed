import numpy as np
import pandas as pd
import fractal_dim as fd
import embed_parse as ep

def getEmbeddingsFromFiles(filenames):
    return map(ep.parseEmbeddingWithMetadata, filenames)

def getAllFractalDimDFs(embeddings,sampleRatio,initRad,radFactor,radCount):
    dfs = []
    for embedding in embeddings:
        singledf = getFractalDimDF(embedding,sampleRatio,initRad,radFactor,radCount)
        dfs.append(singledf)
    return pd.concat(dfs)

def getFractalDimDF(embedding,sampleRatio,initRad,radFactor,radCount):
    """
    Takes in a rich embedding and computes its fractal dimension, storing
    the result along with the embedding info.
    """
    radii,logs = fd.globalFractalDimension(embedding.embedding.T,sampleRatio,initRad,radFactor,radCount,'')
    df = pd.DataFrame.from_dict({'tool':embedding.tool,
                                 'corpus':embedding.corpus,
                                 'replicate':embedding.replicate,
                                 'n-dimension':embedding.ndim,
                                 'radii': radii,
                                 'fractal-dimensions': logs})
    return df


