# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset = pd.read_csv(r"C:\Users\Prasenjit\Desktop\Datasets\Churn_Modelling.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,13].values

#categorical data
#encode the codes into number
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categories = 'auto')
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y =LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#splitting the datasets into training set and test sets 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y,test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)
                                            

