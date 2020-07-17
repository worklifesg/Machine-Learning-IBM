import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Loading Data ##
col_names=['Pregnant','Glucose','BP','Skin','Insulin','BMI','Pedigree','Age','Label']
df=pd.read_csv('D:\Python\edx\Machine Learning\Logistic_Regression\datasets_228_482_diabetes.csv',
               header=None,
               names=col_names)
with open('LogReg.txt','a') as f:
    print(df.head(),file=f)

## Selecting Features ##

f_col=['Pregnant','Glucose','BP','Skin','Insulin','BMI','Pedigree','Age']
X=df[f_col]
y=df.Label

## Splitting Data ##

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

## Model Development and Prediction ##

from sklearn.linear_model import LogisticRegression

lreg=LogisticRegression()
lreg.fit(X_train,y_train)

y_prediction=lreg.predict(X_test)

## Model Evaluation using confusion matrix ##

from sklearn import metrics
c_mat=metrics.confusion_matrix(y_test,y_prediction)
with open('LogReg.txt','a') as f:
    print(c_mat,file=f)

## Data Visualization ##

cnames=[0,1]
fig, ax = plt.subplots()
tmarks=np.arange(len(cnames))
plt.xticks(tmarks,cnames)
plt.yticks(tmarks,cnames)

sns.heatmap(pd.DataFrame(c_mat),annot=True,cmap='YlGnBu',fmt='g')
ax.xaxis.set_label_position('top')

plt.tight_layout()
plt.title('C-Matrix',y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Pedicated Label')

## C matrix evaluation parameters ##
with open('LogReg.txt','a') as f:
    print('Accuracy: ',metrics.accuracy_score(y_test,y_prediction),file=f)
    print('Precision: ',metrics.precision_score(y_test,y_prediction),file=f)
    print('Recall: ',metrics.recall_score(y_test,y_prediction),file=f)

## ROC- Recicver Operating Characteristics curve ##
fig=plt.figure()
y_pred_proba = lreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
