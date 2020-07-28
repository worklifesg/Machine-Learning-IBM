import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

## Load data ##

cdf=pd.read_csv('D:\Python\edx\Machine Learning\Clustering\Cust_Segmentation.csv')
with open('Cust_Kmeans.txt','a') as f:
    print(cdf.head(),file=f)

# Preprocessing # - Address column is categorical and is not needed for kmeans algorithm
df=cdf.drop('Address',axis=1)
with open('Cust_Kmeans.txt','a') as f:
    print(df.head(),file=f)

# Normalizing #

from sklearn.preprocessing import StandardScaler
X=df.values[:,1:]
X=np.nan_to_num(X)
d_set=StandardScaler().fit_transform(X)
with open('Cust_Kmeans.txt','a') as f:
    print(d_set,file=f)

#Modeling#
k_means=KMeans(init='k-means++',n_clusters=3,n_init=12)
k_means.fit(X)
k_means_labels=k_means.labels_ #labels for each point in the model using KMeans' .labels_ attribute
k_means_cluster_centers = k_means.cluster_centers_ #coordinates of the cluster centers using KMeans' .cluster_centers_ 
with open('Cust_Kmeans.txt','a') as f:
    print(k_means_cluster_centers,file=f)
    print(k_means_labels,file=f)

## Insights ##
df['Clus_km']=k_means_labels #Assigning labels to each row in DF
with open('Cust_Kmeans.txt','a') as f:
    print(df.head(),file=f)

df.groupby('Clus_km').mean() # centroid values by avg features in each cluster

# Distribution of customers based on their age and income #
area=np.pi*(X[:,1])**2
plt.scatter(X[:,0],X[:,3],s=area,c=k_means_labels.astype(np.float),alpha=0.5)
plt.xlabel('Age',fontsize=18)
plt.ylabel('Income',fontsize=16)

# 3D plot 
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(1,figsize=(8,6))
plt.clf()
ax=Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:,1],X[:,0],X[:,3],c=k_means_labels.astype(np.float))

#Display plot
plt.show()
