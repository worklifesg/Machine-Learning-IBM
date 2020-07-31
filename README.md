# Machine-Learning-IBM
## Table of contents
* [General info](#general-info)
* [Code setup Information](#code-setup-information)
  * [Simple and Multiple Linear Regression](#simple-and-multiple-linear-regression)
  * [Ordinal Logistic Regression](#ordinal-logistic-regression)
  * [Non-Linear Regression](#non-linear-regression)
* [Classification](#classification)
* [Clustering](#clustering)
## General info
This repository is on Machine Learning using Python 3.8.3 using Visual Studio Code. Most of the programs are from IBM Machine Learning course and some algorithms (course out of scope) are presenterd only for learning purpose. These codes will be on topics like Regression, Classification, Clustering and Recommender Systems. 

## Code setup Information

* The code is writin in Visual Studio code using Python extension with Python 3.8.3 installed on the system. 
* Output results are written in text file (.txt)
* The illustrations are saved as eps and using latex, each documentation is created as .pdf

### Simple and Multiple Linear Regression

* Python files: Simple_Regression_example.py, Multiple_Regression.py
* Date file: FuelConsumptionCo2.csv
* A fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. Source: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
* Output file: LinearReg.txt, MultipleReg.txt

### Ordinal Logistic Regression

* Python files: Ordinal_Regression_example.py
* Date file: datasets_228_482_diabetes.csv
* Pima Indian Diabetes dataset. Source: https://www.kaggle.com/uciml/pima-indians-diabetes-database
* Output file: LogReg.txt

### Non-Linear Regression

* Python files: Non_Linear.py, Non_Linear_Regression.py
* Date file: china_gdp.csv
* The dataset corresponding to China's GDP from 1960 to 2014. This dataset has two columns, the first, a year between 1960 and 2014, the second, China's corresponding annual gross domestic income in US dollars for that year. 
* Output file: NonLinearReg.txt


## Classification

* Separate folder '[Classification](#classification)' for working on different (categorical) datasets and applying different Classification algorithms such as KNN, Decision Tree, Logisitc Regression and Support Vector Machine (SVM)

## Clustering

* Separate folder '[Clustering](#clustering)' for working on different datasets and applying different Clustering algorithms such as K-means (Partitioning Clustering), Agglomerative (Hierarchical Clustering) and DBSCAN (Density Based Clustering)

## Usage of basemap in DBSCAN Clustering Algorithm

Currently basemap is no longer in use in matplotlib toolkits since 2017. Another feature has been introduced 'Cartopy' but to finish this excerise in Visual Studio Code using basemap package, following steps are done.

* Install [Python 2.7.18](https://www.python.org/downloads/release/python-2718/), where pip is already there as there are no pip files before 2.7.9 and needs to be bootstrapped if used.
* Install cp27-amd64 or cp27-amd32 .whl files for [Pyproj](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyproj) and [Basemap](https://www.lfd.uci.edu/~gohlke/pythonlibs/#basemap)
* Using command prompt, use ' pip install pyproj‑1.9.6‑cp27‑cp27m‑win_amd64.whl' and ' pip install basemap‑1.2.1‑cp27‑cp27m‑win_amd64.whl'
* Either you can make python 2.7 as default version in user and system environment variables in system settings or can Select Interpreter in VS Code and choose Python 2.7
* Please remember when installing any version of Python, all other related packages need to be installed again such as numpy, pandas, scikit-learn, scipy etc.

------------------------------------------------------------------------------------------------
Note: Each program is compiled along with output log file and results in pdf using latex.
