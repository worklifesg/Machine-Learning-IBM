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

from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
with open('Log_Reg.txt','a') as f:
    print('J Score: ', jaccard_score(y_test,yhat,labels=None, average='binary', sample_weight=None),file=f)
    print('Accuracy Score: ', accuracy_score(y_test,yhat),file=f)

#Note: In current version of scikit-learn 0.23.1 jaccard_similarity_score is replaced by jaccard_score which
#differs by definition as jaccard_similarity_score is just same as accuracy_score but by definition of 
#jaccard index accuracy score and jaccard score are different. SO, if using scikit-learn > 0.23.1, jaccard_similarity_score not available

#Using Confusion matrix #

from sklearn.metrics import classification_report,confusion_matrix
import itertools

def plot_cmat(cm,classes,normalize=False,
                title='Confusion Matrix',cmap=plt.cm.Blues):
                if normalize:
                    cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
                    print('Normalized Confusion Matrix')
                else:
                    print('COnfusion matrix without Normalization')
                
                with open('Log_Reg.txt','a') as f:
                    print(cm,file=f)

                plt.imshow(cm,interpolation='nearest',cmap=cmap)
                plt.title(title)
                plt.colorbar()
                tick_marks=np.arange(len(classes))
                plt.xticks(tick_marks,classes,rotation=45)
                plt.yticks(tick_marks,classes)

                #For Labeling inside boxes #
                fmt='.2f' if normalize else 'd'
                threshold=cm.max()/2
                for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
                    plt.text(j,i,format(cm[i,j],fmt),
                            horizontalalignment='center',
                            color='white' if cm[i,j] > threshold else 'black')
                plt.tight_layout
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')

#Confusion matrix #
c_mat=confusion_matrix(y_test,yhat,labels=[1,0])
with open('Log_Reg.txt','a') as f:
    print('Confusion matrix: \n ',c_mat,file=f)

#Confusion matrix plot#
plt.figure()
plot_cmat(c_mat,classes=['churn=1','churn=0'],normalize=False,title='Confusion Matrix')
plt.show()

# Compute classification report - Precision, Recall, F1Score and Support

with open('Log_Reg.txt','a') as f:
    print('Classification Report: \n ',classification_report(y_test,yhat),file=f)
