# Clustering
* K-Means is an algorithm for unsupervised machine learning. It is one of the Partitioning Algorithm technique that divides the data into non overlapping subsets (labels) without any cluster-internal structure or labels.
* Agglomerative Clustering - is one of the Hierarchical Clustering used to build hierarchy of clusters where each node is a cluster consisting of clusters of its daughter modes. This approach is bottom up approach, where each observation or data point starts its own cluster and pairs of clusters merge together as they move upwards in the hierarchy.

## Table of contents
* [K-Means](#k-means)
* [Agglomerative Hierarchical Clustering](#agglomerative-hierarchical-clustering)

### K-Means

* Python files: Kmeans_Algorithm.py, Clustering_kmeans_CustomerSegmentation.py
* Date file: Cust_Segmentation.csv
* Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources. For example, one group might contain customers who are high-profit and low-risk, that is, more likely to purchase products, or subscribe for a service. A business task is to retaining those customers. Another group might include customers from non-profit organizations. And so on. 
* Two programs:
  * General K-Means Algorithm for randomly generated data points - Kmeans_Algorithm.py
  * K-Means Algorithm for customer segmentation dataset - Clustering_kmeans_CustomerSegmentation.py
* Output file: Kmeans_general.txt , Cust_Kmeans.txt

### Agglomerative Hierarchical Clustering

* Python files: Hierarchical_Clustering_Agglomerative.py, Hierarchical_Clustering_Vehicle_Dataset_Scipy.py,Hierarchical_Clustering_Vehicle_Dataset_Scikit.py
* Date file: cars_clus.csv
* Automobile manufacturer has developed prototypes for a new vehicle. Before introducing the new model into its range, the manufacturer wants to determine which existing vehicles on the market are most like the prototypes--that is, how vehicles can be grouped, which group is the most similar with the model, and therefore which models they will be competing against. The objective here, is to use clustering methods, to find the most distinctive clusters of vehicles. It will summarize the existing vehicles and help manufacturers to make decision about the supply of new models.
* Three programs:
  * General Agglomerative Clustering Algorithm for randomly generated data points - Hierarchical_Clustering_Agglomerative.py
  * Agglomerative Clustering Algorithm for vehicle dataset using scipy - Hierarchical_Clustering_Vehicle_Dataset_Scipy.py
  * Agglomerative Clustering Algorithm for vehicle dataset using scikit-learn - Hierarchical_Clustering_Vehicle_Dataset_Scikit.py
* Output file: Hierarchical.txt , hierarchical_vehicle.txt, hierarchical_vehicle_s.txt
