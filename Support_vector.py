# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:57:18 2020

@author: Prasenjit
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\Prasenjit\Desktop\Python_Files\SVR\Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting SVR to Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)



#Predicting the Test Set results
y_pred = regressor.predict(6.5)

# Visulising the training SVR results 
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor.predict(X), color = 'blue' )
plt.title('Salary vs Experience')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

