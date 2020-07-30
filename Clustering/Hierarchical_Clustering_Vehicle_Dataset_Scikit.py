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
with open('hierarchical_vehicle_s.txt','a') as f:
    print(df.head(),file=f)
    print(df.shape,file=f)

## Data Cleaning ## clear the dataset by dropping the rows that have null value:

with open('hierarchical_vehicle_s.txt','a') as f:
    print('Shape of data set before cleaning: ',df.size,file=f)

df[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')

df=df.dropna()
df=df.reset_index(drop=True)
with open ('hierarchical_vehicle_s.txt','a') as f:
    print('Shape of the dataset after cleaning: ',df.size,file=f)
    print(df.head(5),file=f)

#Feature set
feat_set=df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Normalization - between 0,1 for each feature using MinMaxScalar

from sklearn.preprocessing import MinMaxScaler
x=feat_set.values
min_max=MinMaxScaler()
feat_matrix=min_max.fit_transform(x)
with open ('hierarchical_vehicle_s.txt','a') as f:
    print(feat_matrix[0:5],file=f)

# Second method - Clustering using scikit-learn
d_mat=distance_matrix(feat_matrix,feat_matrix)
with open ('hierarchical_vehicle_s.txt','a') as f:
    print(d_mat,file=f)

#AgglomerativeClustering performs a hierarchical clustering using a bottom up approach. 
# The linkage criteria determines the metric used for the merge strategy:
# Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing
# approach and in this sense is similar to the k-means objective function but tackled with an 
# agglomerative hierarchical approach.
# Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
# Average linkage minimizes the average of the distances between all observations of pairs of clusters.

agglom=AgglomerativeClustering(n_clusters=6,linkage='complete')
agglom.fit(feat_matrix)
with open ('hierarchical_vehicle_s.txt','a') as f:
    print(agglom.labels_,file=f)

# Adding new column - cluster to the data
df['cluster_']=agglom.labels_
with open ('hierarchical_vehicle_s.txt','a') as f:
    print(df.head(),file=f)

## Plotting scatter plot for data points with their clusters
import matplotlib.cm as cm
n_clusters=max(agglom.labels_)+1
colors=cm.rainbow(np.linspace(0,1,n_clusters))
cluster_labels=list(range(0,n_clusters))

plt.figure(figsize=(16,14))
for color, label in zip(colors,cluster_labels):
    subset=df[df.cluster_==label]
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

 # Centroids of each cluster is not clear in scatter plot, so we can summarize first
 #classes and then the clusters. There are two classes - Cars and Trucks

qdf=df.groupby(['cluster_','type'])['cluster_'].count()
with open ('hierarchical_vehicle_s.txt','a') as f:
    print(qdf,file=f)

#For characteristics of each cluster
agg_cars=df.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
with open ('hierarchical_vehicle_s.txt','a') as f:
    print(agg_cars,file=f)

##It is obvious that we have 3 main clusters with the majority of vehicles in those.
##Cars:
    ##Cluster 1: with almost high mpg, and low in horsepower.
    ##Cluster 2: with good mpg and horsepower, but higher price than average.
   ## Cluster 3: with low mpg, high horsepower, highest price.
##Trucks:
    ##Cluster 1: with almost highest mpg among trucks, and lowest in horsepower and price.
    ##Cluster 2: with almost low mpg and medium horsepower, but higher price than average.
    ##Cluster 3: with good mpg and horsepower, low price.

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

#Display plot
plt.show()
