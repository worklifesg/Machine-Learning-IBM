# Classification
* K-Nearest Neighbors is an algorithm for supervised machine learning. In this algorithm, once a point is to be predicted, it takes into account the 'K' nearest points to it to determine it's classification. 
## Table of contents
* [K-Nearest Neighbors](#k-nearest-neighbors)
* [Decision Tree using Scikit-learn](#decision-tree-using-scikit-learn)
* [Logistic Regression](#logistic-regression)

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
