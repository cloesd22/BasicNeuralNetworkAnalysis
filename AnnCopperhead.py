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
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=1)


from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scY = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)
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


def Build_Ann():
    classifier = Sequential()
    classifier.add(Dense(output_dim=10,init='uniform',activation='relu',input_dim=5))
    classifier.add(Dense(output_dim=7,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=4,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

#classifier = KerasClassifier(build_fn=Build_Ann,epochs=500,batch_size=25,verbose=1)
#accuracies = cross_val_score(estimator=classifier,X=X_train,y=Y_train,n_jobs=1,cv=10)
#mean=accuracies.mean()
#variance = accuracies.std()
##    
classifier = Build_Ann()
classifier.fit(X_train,Y_train,batch_size= 10,epochs=1000)
#X_sampletest = np.array([[41,1,0,2]])
Y_sampletestpred = classifier.predict(X_test)
Y_sampletestpred = (Y_sampletestpred>0.2)

#
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_sampletestpred)    
#        

