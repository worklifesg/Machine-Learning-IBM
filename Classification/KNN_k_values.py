#Objective is to find best value of K in KNN classifier to achieve maximum accuracy

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

#1- Basic Service 2- E-Service 3- Plus Service 4- Total Service#
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

#Finding the value of k to achieve maximum accuracy via Confusion matrix and then use the value of k in the program#
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks=10
acc_mean=np.zeros((Ks-1))
acc_std=np.zeros((Ks-1))
Cmat=[]
for n in range(1,Ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    acc_mean[n-1]=metrics.accuracy_score(y_test,yhat)

    acc_std[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
with open('KNN_kvalue.txt','a') as f:
    print(acc_mean,file=f)
    print('The best accuracy was with: ', acc_mean.max(), 'with k= ',acc_mean.argmax()+1,file=f)

#Plot of k values vs accuracy #
plt.figure()
plt.plot(range(1,Ks),acc_mean,'g')
plt.fill_between(range(1,Ks),acc_mean - 1 * acc_std,acc_mean + 1 * acc_std, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ',fontname='Times New Roman',fontsize=12)
plt.xlabel('Number of Nabors (K)',fontname='Times New Roman',fontsize=12)
plt.tight_layout()
plt.show()
