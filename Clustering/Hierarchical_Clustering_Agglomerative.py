# To perform agglomerative clustering on randomly generated data

import numpy as np
import pandas as pd 
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt 
from sklearn import manifold,datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

# Generating random data #

## n_samples= total data points, centers= no. of centers to generate, cluster_std=standard deviation of clusters

X1,y1=make_blobs(n_samples=50,centers=[[4,4],[-2,-1],[1,1],[10,4]],cluster_std=0.9)
#Scatter plot of randomly generated data
plt.scatter(X1[:,0],X1[:,1],marker='o')

# Agglomerative Clustering #
#n_clusters= no. of clusters to form along with their centroids, linkae - criterion:complete,average

agg=AgglomerativeClustering(n_clusters=4,linkage='average')
agg.fit(X1,y1)

# Plotting after clustering
plt.figure(figsize=(6,4))
x_min,x_max=np.min(X1,axis=0),np.max(X1,axis=0)
X1=(X1-x_min)/(x_max-x_min)

for i in range(X1.shape[0]):
    plt.text(X1[i,0],X1[i,1],str(y1[i])),
    color=plt.cm.nipy_spectral(agg.labels_[i]/10),
    fontdict={'weight':'bold','size':9}

plt.xticks([])
plt.yticks([])

plt.scatter(X1[:,0],X1[:,1],marker='.')

## Dendrogram ##

dmat=distance_matrix(X1,X1)
with open('Hierarchical.txt','a') as f:
    print(dmat,file=f)
Z = hierarchy.linkage(dmat, 'complete')
plt.figure()
dendro = hierarchy.dendrogram(Z)
# Display plot
plt.show()
