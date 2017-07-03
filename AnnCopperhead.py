# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:23:34 2017

@author: manoj
"""

#Ann

import numpy as np
import matplotlib as mtpltlb
import pandas as pd

#E:\SnakeGarden\Copperhead\Copperhead\Copperhead\spiders

#Data Preprocessing
#===============================================
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,2:7]
Y = dataset.iloc[:,8]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.0,random_state=0)
#SET TEST_SIZE TO 0 TO TURN ENTIRE DATASET INTO TRAINING SET FOR K-CROSSVALIDATION


from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scY = StandardScaler()
X_train = scX.fit_transform(X_train)
#X_test = scX.transform(X_test)
#Y_train = scY.fit_transform(Y_train)
#Y_test = scY.transform(Y_test)

#===============================================


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# BUILD NEURAL NETWORK ARCHITECTURE
def Build_Ann():
    classifier = Sequential()
    classifier.add(Dense(output_dim=5,init='uniform',activation='relu',input_dim=5))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(output_dim=5,init='uniform',activation='relu'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier



    
#NETWORK TRAINING
classifier = Build_Ann()
classifier.fit(X_train,Y_train,batch_size= 25,epochs=2500)

## SELECT ONE OF THE FOLLOWING:

#CROSS VALIDATION OF STRUCTURE BASED ON INPUT DATA.
#cross validation on entire data set, producing avg accuracy and variance
#classifier = KerasClassifier(build_fn=Build_Ann,epochs=125,batch_size=22,verbose=1)
#accuracies = cross_val_score(estimator=classifier,X=X_train,y=Y_train,n_jobs=1,cv=10)
#mean=accuracies.mean()
#variance = accuracies.std()



#SINGLE TEST
##Single test on 1 data point, input X, it will predict and output Y
#X_sampletest = np.array([[103,3,1,1,4.63]])
#X_sampletest = scX.transform(X_sampletest)
#Y_sampletestpred = classifier.predict(X_sampletest)
##THRESHOLD SETUP
#Y_sampletestpred = (Y_sampletestpred>0.25)


#TEST SET FULL TEST
##Inputs full X_test dataset into neural network, and produces confusion matrix to evaluate.
Y_sampletestpred = classifier.predict(X_train)
Y_sampletestpred = (Y_sampletestpred>0.25)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, Y_sampletestpred)    
        

