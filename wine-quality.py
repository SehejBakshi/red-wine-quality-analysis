import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn

df=pd.read_csv('winequality-red.csv')
print(df.head())

count=0
for i in df.isnull().sum(axis=1):
    if i>0:
        count+=1

print('Total no of rows with missing values:', count)

print(df.columns)

x=df[['fixed acidity', 'volatile acidity',
      'citric acid', 'residual sugar',
      'chlorides', 'free sulfur dioxide',
      'total sulfur dioxide', 'density',
      'pH', 'sulphates', 'alcohol', 'quality']]
y=df[['quality']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

print('Training set shape:', x_train.shape, y_train.shape)
print('Testing set shape:', x_test.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)

ypred=lr.predict(x_test)
print('Accuracy of model:', sklearn.metrics.accuracy_score(y_test, ypred))
