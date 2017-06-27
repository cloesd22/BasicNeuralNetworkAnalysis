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
Y = dataset.iloc[:,7]

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
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
Y_train=sc_Y.fit_transform(Y_train)
Y_test=sc_Y.fit(Y_train)

#Data pre-processing=====================================================



#Building ANN Structure==================================================
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=5))
classifier.add(Dense(output_dim=4,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=10, epochs=500)


#Building ANN Structure==================================================
for layer in classifier.layers:
    weights=layer.get_weights()