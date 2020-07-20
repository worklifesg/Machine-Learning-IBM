#Objective is to build classifier and predict class of unknown classes using K-nearest neighbour (KNN)

import itertools
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import matplotlib.ticker as ticker

from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
from matplotlib import rc,font_manager

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal',
    size=12, weight='normal', stretch='normal')
ax=plt.gca()

## Loading Data ##
df=pd.read_csv('D:\Python\edx\Machine Learning\Classification\elecom_customer_data.csv')
with open('KNN.txt','a') as f:
    print(df.head(),file=f)
    print(df.describe(),file=f)
    print('Classes are: ', df['custcat'].value_counts(),file=f)

#1- Basic Service 2- E-Service 3- Plus Service 4- Total Service#


df.hist(column='income',bins=50)

##Feature Set X,Y##

print(df.columns)
X=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside',]].values
X[0:5]

y=df['custcat'].values
y[0:5]

# Normalize Data #
X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

# Train Test Split#
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
with open('KNN.txt','a') as f:
    print('Train set: ', X_train.shape,y_train.shape,file=f)
    print('Test set: ', X_test.shape,y_test.shape,file=f)

# Classification - K nearest neighbour#

from sklearn.neighbors import KNeighborsClassifier

k=9 # Train model and predict with k=9 (best value for high accuracy, check KNN_k_values.py program)
neigh=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
yhat=neigh.predict(X_test)
with open('KNN.txt','a') as f:
    print(yhat[0:5],file=f)

#Predict using accuracy_score which is similar to jaccard_similarity_score function#

from sklearn import metrics
with open('KNN.txt','a') as f:
    print('Train set accuracy: ', metrics.accuracy_score(y_train,neigh.predict(X_train)),file=f)
    print('Test set accuracy: ',metrics.accuracy_score(y_test,yhat),file=f)
plt.show()
