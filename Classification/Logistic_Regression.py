# In progress. Not completed yet.

import pandas as pd 
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from matplotlib import rc,font_manager

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal',
    size=12, weight='normal', stretch='normal')
ax=plt.gca()

## Loading Data ##
df=pd.read_csv('D:\Python\edx\Machine Learning\Classification\ChurnData.csv')
with open('Log_Reg.txt','a') as f:
    print(df.head(),file=f)

## Preprocessing and selection ##
df=df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
df['churn']=df['churn'].astype('int')
with open('Log_Reg.txt','a') as f:
    print(df.head(),file=f)
    print(df.shape,file=f)
for col in df.columns:
    with open('Log_Reg.txt','a') as f:
        print(col,file=f)

## Define X,y dataset ##
X=np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y=np.asarray(df['churn'])
with open('Log_Reg.txt','a') as f:
    print(X[0:5],file=f)
    print(y[0:5],file=f)

## Normalize dataset ##
X=preprocessing.StandardScaler().fit(X).transform(X)
with open('Log_Reg.txt','a') as f:
    print(X[0:5],file=f)

## Train_Test_Split ##
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=4)
with open('Log_Reg.txt','a') as f:
    print('Train set: ', X_train.shape,y_train.shape,file=f)
    print('Test set: ', X_test.shape,y_test.shape,file=f)

## Modeling using scikit-learn ##
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LogReg=LogisticRegression(C=0.01,solver='liblinear').fit(X_train,y_train) # C-inverse of regularization strength (positive float)

yhat=LogReg.predict(X_test)
yhat_prob=LogReg.predict_proba(X_test) # predict_proba returns estimates for all classes, ordered by the label of classes

### Evaluation ##

from sklearn.metrics import jaccard_similarity_score
with open('Log_Reg.txt','a') as f:
    print('J Score: ', jaccard_similarity_score(y_test,yhat),file=f)
