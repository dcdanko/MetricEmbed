
from embed_parse import *
from metric_entropy_vec import * 
from glob import glob


embeddingfiles = glob('Embeddings/*brown*.txt')
print(embeddingfiles)
embeddings = [parseEmbeddingWithMetadata(embf) for embf in embeddingfiles]
mes = getAllMetricEntropyDFs(embeddings,[0.025,0.05,0.1,0.2,0.4],pairMetric='cosine') 

