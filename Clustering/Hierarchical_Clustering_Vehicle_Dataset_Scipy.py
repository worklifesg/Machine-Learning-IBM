#Agglomerative Clustering using Scipy

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn import manifold,datasets
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster

## Load data ##
df=pd.read_csv('D:\Python\edx\Machine Learning\Clustering\cars_clus.csv')
with open('hierarchical_vehicle.txt','a') as f:
    print(df.head(),file=f)
    print(df.shape,file=f)

## Data Cleaning ## clear the dataset by dropping the rows that have null value:

with open('hierarchical_vehicle.txt','a') as f:
    print('Shape of data set before cleaning: ',df.size,file=f)

df[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')

df=df.dropna()
df=df.reset_index(drop=True)
with open ('hierarchical_vehicle.txt','a') as f:
    print('Shape of the dataset after cleaning: ',df.size,file=f)
    print(df.head(5),file=f)

#Feature set
feat_set=df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Normalization - between 0,1 for each feature using MinMaxScalar

from sklearn.preprocessing import MinMaxScaler
x=feat_set.values
min_max=MinMaxScaler()
feat_matrix=min_max.fit_transform(x)
with open ('hierarchical_vehicle.txt','a') as f:
    print(feat_matrix[0:5],file=f)

# First method - Clustering using Scipy
##In agglomerative clustering, at each iteration, the algorithm must update the 
# distance matrix to reflect the distance of the newly formed cluster with the remaining 
# clusters in the forest. The following methods are supported in Scipy for calculating 
# the distance between the newly formed cluster and each: 
# - single - complete - average - weighted - centroid

import scipy
leng=feat_matrix.shape[0]
D=scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j]=scipy.spatial.distance.euclidean(feat_matrix[i],feat_matrix[j])

import pylab
Z=hierarchy.linkage(D,'complete')

# for paritioning in clustering we draw a cutting line
max_d=3
clusters=fcluster(Z,max_d,criterion='distance')
k=5
clusters_max=fcluster(Z,k,criterion='maxclust')
with open ('hierarchical_vehicle.txt','a') as f:
    print(clusters,file=f)
    print(clusters_max,file=f)

# Dendrogram
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (df['manufact'][id], df['model'][id], int(float(df['type'][id])) )

dendro=hierarchy.dendrogram(Z,leaf_label_func=llf, leaf_rotation=0, leaf_font_size =4, orientation = 'right')

#Display plot
plt.show()
