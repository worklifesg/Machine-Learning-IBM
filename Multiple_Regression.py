import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, font_manager
from sklearn import linear_model

ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal',
    size=12, weight='normal', stretch='normal')
plt.style.use('seaborn-white')
ax=plt.gca()

## Loading Data ##
df=pd.read_csv('D:\Python\edx\Machine Learning\FuelConsumptionCo2.csv')
with open('MultipleReg.txt','a') as f:
    print(df.head(),file=f)
    print(df.describe(),file=f)

## Data features to be used for regression ##

f_col=['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']
X=df[f_col]
with open('MultipleReg.txt','a') as f:
    print(X.head(9),file=f)

plt.figure()
plt.scatter(X.ENGINESIZE,X.CO2EMISSIONS,color='blue')
plt.title('Scatter Plot - Engine Size vs Emissions',fontname='Times New Roman',
          fontsize=12)
plt.ylabel('Emissions',fontname='Times New Roman',fontsize=12)
plt.xlabel('Engine Size',fontname='Times New Roman',fontsize=12)

## Train Test Data ##

mask=np.random.rand(len(df))<0.8
train=X[mask]
test=X[mask]

plt.figure()
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.title('Train Data Plot - Engine Size vs Emissions',fontname='Times New Roman',
          fontsize=12)
plt.ylabel('Emissions',fontname='Times New Roman',fontsize=12)
plt.xlabel('Engine Size',fontname='Times New Roman',fontsize=12)

## MLR ## uses Ordinary Least Square (OLS) OLS can find the best parameters using of the following methods:
#  - Solving the model parameters analytically using closed-form equations - 
# Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)

Lreg=linear_model.LinearRegression()
x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(train[['CO2EMISSIONS']])
Lreg.fit(x,y)
with open('MultipleReg.txt','a') as f:
    print('Coefficients: ', Lreg.coef_,file=f)
    print('Intercept: ',Lreg.intercept_,file=f)

## Prediction ##

y_hat=Lreg.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x1=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y1=np.asanyarray(test[['CO2EMISSIONS']])

with open('MultipleReg.txt','a') as f:
    print('Residual Sum of squares: %.2f'%np.mean((y_hat-y)**2),file=f)
    print('Variance score: %.2f'%Lreg.score(x1,y1),file=f)
## Display Plot ##
plt.show()