# Classification
* K-Nearest Neighbors is an algorithm for supervised machine learning. In this algorithm, once a point is to be predicted, it takes into account the 'K' nearest points to it to determine it's classification. 
* Decision Tree algorithm maps all possible decisions path in the form of a tree by choosing the attributes that returns highest information gain. 
* Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable, y, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.
* SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong.
## Table of contents
* [K-Nearest Neighbors](#k-nearest-neighbors)
* [Decision Tree using Scikit-learn](#decision-tree-using-scikit-learn)
* [Logistic Regression](#logistic-regression)
* [Support Vector Machine (SVM)](#support-vector-machine-(SVM))

### K-Nearest Neighbors

* Python files: KNN_example.py, KNN_k_values.py
* Date file: elecom_customer_data.csv
* Telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. The target field, called custcat, has four possible values that correspond to the four customer groups, as follows: 1- Basic Service 2- E-Service 3- Plus Service 4- Total Service. Our objective is to build a classifier, to predict the class of unknown cases. We will use a specific type of classification called K nearest neighbour.
* Two python files: One for finding K value for highest accuracy achievable and another one for KNN classifier algorithm.
* Output file: KNN.txt, KNN_kvalue.txt

### Decision Tree using Scikit-learn

* Python files: Decision_Tree_Example.py
* Date file: drug200.csv
* Dataset of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y. The objective is to find out which drug might be appropriate for a future patient with the same illness. The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the drug that each patient responded to. 
* Output file: Decision_Tree.txt

### Logistic Regression

* Python files: Logistic_Regression.py
* Date file: ChurnData.csv
* Telecommunications dataset for predicting customer churn. The dataset includes information about:
    * Customers who left within the last month – the column is called Churn.
    * Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
    * Customer account information – how long they had been a customer, contract, payment method, paperless billing, monthly charges, and total charges
    * Demographic info about customers – gender, age range, and if they have partners and dependents
* Output file: Log_Reg.txt

### Support Vector Machine (SVM)

* Python files: SVM_Example.py
* Date file: cell_samples.csv
* Cancer Data Classification using SVM:
    * Here will use SVM (Support Vector Machines) to build and train a model using human cell records, and classify cells to whether the samples are benign or malignant.
    * The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html].
    The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics.
* Output file: SVM.txt
