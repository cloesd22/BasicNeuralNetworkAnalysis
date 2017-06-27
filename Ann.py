# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:24:32 2017

@author: manoj
"""

#ANN

import numpy as np
import matplotlib as mtpltlib
import pandas as pd


#Data pre-processing=====================================================

#Import data set up X and Y variables
dataset = pd.read_csv('sampledata.csv',encoding = "ISO-8859-1")
X = dataset.iloc[:,2:7]
Y = dataset.iloc[:,8]


#no need to use label encoder as all variables are numeric
#no need to use 1 hot encoder as all varaibles are numeric ((No need for 1ofK))

#Seperate into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#scale the data
#Note, 2 standard scaler objects, one for each axis. training is fit+transform,
#while test is just transform.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Data pre-processing=====================================================



#Building ANN Structure==================================================
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=3,init='uniform',activation='relu',input_dim=5))
    classifier.add(Dense(output_dim=3,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


#Building ANN Structure==================================================

#Cross-Validation==================================================

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10,n_jobs=1)
#Remmeber, n_jobs =1, if <1 or >1 wit will fail.
mean=accuracies.mean()
variance = accuracies.std()


#Cross-Validation==================================================

