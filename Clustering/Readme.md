# Clustering
* K-Means is an algorithm for unsupervised machine learning. It is one of the Partitioning Algorithm technique that divides the data into non overlapping subsets (labels) without any cluster-internal structure or labels.
* Agglomerative Clustering - is one of the Hierarchical Clustering used to build hierarchy of clusters where each node is a cluster consisting of clusters of its daughter modes. This approach is bottom up approach, where each observation or data point starts its own cluster and pairs of clusters merge together as they move upwards in the hierarchy.
* DBSCAN Algorithm is Density Based Spatial Clustering of Applications with Noise used for arbitary shape clusters, where tradiation techniques suhc as k-means doesn't differentiate between actual data points and outliers. This algorithm makes more dense areas with respect to their radius and maximum number of data points around centroid and creates less dense areas as outliers.
## Table of contents
* [K-Means](#k-means)
* [Agglomerative Hierarchical Clustering](#agglomerative-hierarchical-clustering)
* [DBSCAN](#dbscan)

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

### DBSCAN Clustering

* Python files: DBSCAN_General.py, DBSCAN_Weather_Station_Clustering.py
* Date file: weather_Canada.csv
* This dataset shows the location of weather stations in Canada. DBSCAN can be used here, for instance, to find the group of stations which show the same weather condition. As you can see, it not only finds different arbitrary shaped clusters, can find the denser part of data-centered samples by ignoring less-dense areas or noises.
* Two programs:
  * General DBSCAN Algorithm for randomly generated data points - DBSCAN_General.py
  * DBSCAN Algorithm for weather stations dataset - DBSCAN_Weather_Station_Clustering.py
* Output file: DBSCAN_general.txt

## Usage of basemap in DBSCAN algorithm

Currently basemap is no longer in use in matplotlib toolkits since 2017. Another feature has been introduced 'Cartopy' but to finish this excerise in Visual Studio Code using basemap package, following steps are done.

* Install [Python 2.7.18](https://www.python.org/downloads/release/python-2718/), where pip is already there as there are no pip files before 2.7.9 and needs to be bootstrapped if used.
* Install cp27-amd64 or cp27-amd32 .whl files for [Pyproj](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyproj) and [Basemap](https://www.lfd.uci.edu/~gohlke/pythonlibs/#basemap)
* Using command prompt, use ' pip install pyproj‑1.9.6‑cp27‑cp27m‑win_amd64.whl' and ' pip install basemap‑1.2.1‑cp27‑cp27m‑win_amd64.whl'
* Either you can make python 2.7 as default version in user and system environment variables in system settings or can Select Interpreter in VS Code and choose Python 2.7
* Please remember when installing any version of Python, all other related packages need to be installed again such as numpy, pandas, scikit-learn, scipy etc.
