# Using DBSCAN for random generated data points

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs

# Data generation function #

def datapoints(centroid,samples,clus_deviation):
    X,y=make_blobs(n_samples=samples,centers=centroid,cluster_std=clus_deviation)
    X=StandardScaler().fit_transform(X)
    return X,y

X, y = datapoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)

# Modeling #
# DBSCAN works based on two parameters: Epsilon and Minimum Points
# Epsilon determine a specified radius that if includes enough number of points 
# within, we call it dense area 
# minimumSamples determine the minimum number of data points we want in a 
# neighborhood to define a cluster.

eps=0.3
min_samples=7
db=DBSCAN(eps=eps,min_samples=min_samples).fit(X)
labels=db.labels_
with open('DBSCAN_general.txt','a') as f:
    print(labels,file=f)

## Distinguish Outliers ##

samples_mask=np.zeros_like(db.labels_,dtype=bool) # array creation of booleans
samples_mask[db.core_sample_indices_]=True

n_clusters_=len(set(labels))-(1 if -1 in labels else 0) # number of clusters in labels

unique_labels=set(labels) # remove repetition in labels by turning into a set

with open('DBSCAN_general.txt','a') as f:
    print(samples_mask,file=f)
    print(n_clusters_,file=f)
    print(unique_labels,file=f)

## Data visualization ##
colors=plt.cm.Spectral(np.linspace(0,1,len(unique_labels))) # Cluster color creation

for k,col in zip(unique_labels,colors): # color of points
    if k==-1:
        col='k' # Black for noise
    class_mask=(labels==k)

    xy=X[class_mask & samples_mask]
    plt.scatter(xy[:,0],xy[:,1],s=50,c=[col],marker=u'o',alpha=0.5) # Data point plot

    xy=X[class_mask & ~samples_mask]
    plt.scatter(xy[:,0],xy[:,1],s=50,c=[col],marker=u'o',alpha=0.5) # Outliers plot  

# Using K-Means #
from sklearn.cluster import KMeans 
k = 3
k_means3 = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means3.fit(X)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means3.labels_ == k)
    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)

#Display plot
plt.show()
