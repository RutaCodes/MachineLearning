#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:21:03 2022

@author: rutabasijokaite
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def SVR_hyper(x_train: np.array, y_train: np.array):
    
    #Paramater grid used to generate parameter combinations 
    parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.3,0.4,0.5]}
    #Support vector regressor
    svr = SVR()
    #Implementing exhaustive grid search
    clf = GridSearchCV(svr, parameters) 
    clf.fit(x_train,y_train)
    #Combination with the highest cross validation score is retained
    Best_SVR = clf.best_params_
    
    return Best_SVR
