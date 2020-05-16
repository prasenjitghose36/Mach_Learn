# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:58:20 2020

@author: Prasenjit
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Prasenjit\Desktop\Python\Data_Preprocessing\Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
correct = SimpleImputer(missing_values = np.NaN, strategy = "mean")
#now we have to fit the variable correct which contains the object of the class SimpleImputer
correct.fit(X[:,1:3])

#very impportant part is encoding the data in which we have take 
#the string values as the numerical values so the we can perform regression or any other ML on it 
#label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder
X[:,0] = labelencoder_X.fit_transform(X[:,0])
y[:,0] = labelencoder_y.fit_transform(y[:,0])



