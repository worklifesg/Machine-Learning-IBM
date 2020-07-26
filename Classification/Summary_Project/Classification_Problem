import itertools
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib import rc,font_manager
from sklearn import preprocessing
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal',
    size=12, weight='normal', stretch='normal')
ax=plt.gca()

#Loading Data and Data Preprocessing:
df=pd.read_csv('D:\Python\edx\Machine Learning\Classification example\loan_train.csv')
with open('class_problem.txt','a') as f:
    print(df.head(),file=f)
    print(df.shape,file=f)

    ## Convert to date time object

df['due_date']=pd.to_datetime(df['due_date'])
df['effective_date']=pd.to_datetime(df['effective_date'])
with open('class_problem.txt','a') as f:
    print(df.head(),file=f)

# Data visulaization (using seaborn) and Preprocessing

status=df['loan_status'].value_counts()
with open('class_problem.txt','a') as f:
    print(status,file=f)

bins=np.linspace(df.Principal.min(),df.Principal.max(),10)
g=sns.FacetGrid(df,col='Gender',hue='loan_status',palette='Set1',col_wrap=2)
g.map(plt.hist,'Principal',bins=bins,ec='k')
g.axes[-1].legend()

bins1=np.linspace(df.age.min(),df.age.max(),10)
g1=sns.FacetGrid(df,col='Gender',hue='loan_status',palette='Set1',col_wrap=2)
g1.map(plt.hist,'age',bins=bins1,ec='k')
g1.axes[-1].legend()

## Feature Selection and Extraction
# Creating Weekday and weekend columns#

df['dayofweek']=df['effective_date'].dt.dayofweek
bins2=np.linspace(df.dayofweek.min(),df.dayofweek.max(),10)
g2=sns.FacetGrid(df,col='Gender',hue='loan_status',palette='Set1',col_wrap=2)
g2.map(plt.hist,'dayofweek',bins=bins2,ec='k')
g2.axes[-1].legend()
sns.set_style("darkgrid",{'font.sans-serif': ['Arial']})

df['weekend']=df['dayofweek'].apply(lambda x: 1 if (x>3) else 0)
with open('class_problem.txt','a') as f:
    print(df.head(),file=f)

# Converting categorical features to numerical values
### Gender ##
group1=df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
with open('class_problem.txt','a') as f:
    print(group1,file=f)

#Converting male=0 and female=1

df['Gender'].replace(to_replace=['male','female'],value=[0,1],inplace=True)
with open('class_problem.txt','a') as f:
    print(df.head(),file=f)

### Education ##
group2=df.groupby(['education'])['loan_status'].value_counts(normalize=True)
with open('class_problem.txt','a') as f:
    print(group2,file=f)
#Using one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
with open('class_problem.txt','a') as f:
    print(Feature.head(),file=f)

#Mentioning X,y
X=Feature
y=df['loan_status'].values
with open('class_problem.txt','a') as f:
    print(X[0:5],file=f)
    print(y[0:5],file=f)

#Normalizing Data
X = preprocessing.StandardScaler().fit(X).transform(X)
with open('class_problem.txt','a') as f:
    print(X[0:5],file=f)

#####  Classification - KNN, Decision Tree, SVM,LR #####

#Split Data Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=4)
with open('class_problem.txt','a'):
    print('Train set: ',X_train.shape,y_train.shape)
    print('Test set: ', X_test.shape,y_test.shape)

#Modeling
#KNN- first we will find best value of k and then will train the data

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks=15
acc_mean=np.zeros((Ks-1))
acc_std=np.zeros((Ks-1))
Cmat=[]
for n in range(1,Ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat_KNN=neigh.predict(X_test)
    acc_mean[n-1]=metrics.accuracy_score(y_test,yhat_KNN)

    acc_std[n-1]=np.std(yhat_KNN==y_test)/np.sqrt(yhat_KNN.shape[0])
with open('class_problem.txt','a') as f:
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


k=7 #Best value found above
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
DT_model.fit(X_train,y_train)

#SVM
from sklearn import svm
SVM_model = svm.SVC()
SVM_model.fit(X_train, y_train)

#Logitic Regression
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(C=0.01).fit(X_train,y_train)

############ Test Evaluation Data #############

test_df = pd.read_csv('D:\Python\edx\Machine Learning\Classification example\loan_test.csv')
with open('class_problem.txt','a') as f:
    print('Test Data: \n',test_df.head(),file=f)

#Preprocessing for test data as same as train data
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
test_X = preprocessing.StandardScaler().fit(test_Feature).transform(test_Feature)
test_y = test_df['loan_status'].values
with open('class_problem.txt','a') as f:
    print(test_X[0:5],file=f)
    print(test_y[0:5],file=f)

#Evaluating accuracy #
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

knn_yhat = kNN_model.predict(test_X)
DT_yhat = DT_model.predict(test_X)
SVM_yhat = SVM_model.predict(test_X)
LR_yhat = LR_model.predict(test_X)
LR_yhat_prob = LR_model.predict_proba(test_X)

with open('class_problem.txt','a') as f:
    print("KNN Jaccard index: ",  accuracy_score(test_y, knn_yhat),file=f)
    print("KNN F1-score: ",  f1_score(test_y, knn_yhat, average='weighted') ,file=f)
    print("DT Jaccard index: ",  accuracy_score(test_y, DT_yhat),file=f)
    print("DT F1-score: ", f1_score(test_y, DT_yhat, average='weighted') ,file=f)
    print("SVM Jaccard index: ", accuracy_score(test_y, SVM_yhat),file=f)
    print("SVM F1-score: ",f1_score(test_y, SVM_yhat, average='weighted') ,file=f)
    print("LR Jaccard index: f", accuracy_score(test_y, LR_yhat),file=f)
    print("LR F1-score: %.2f", f1_score(test_y, LR_yhat, average='weighted'),file=f )
    print("LR LogLoss: %.2f ", log_loss(test_y, LR_yhat_prob),file=f )

#Display plot
plt.show()
