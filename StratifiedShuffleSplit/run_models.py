#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:56:27 2022

@author: rutabasijokaite
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
#from sklearn import cross_validation
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKfold #ShuffleSplit

def run_models(x_train: pd.DataFrame , y_train: pd.DataFrame):
    
    df = pd.DataFrame(columns=['Model']) 
    models = [
            ('LogReg', LinearRegression()), 
            ('RF', RandomForestRegressor()),
            ('XGB', XGBRegressor()),
            #('SVR',SVR()),
            ('SVR',SVR(kernel="rbf", C=10, gamma="auto")),
           #model1=SVR(kernel="linear", C=100, gamma="auto")
            ('GBC',GradientBoostingRegressor()),
    ]
        
    
    for name, model in models:
        
        x_train_no_nr = np.delete(x_train,0,1)
        cv_custom = StratifiedKfold(n_split=4,shuffle=True)
        scores = cross_validate(model, x_train_no_nr, y_train, groups=x_train[:,0], cv=cv_custom, scoring='r2',return_train_score=True)
        #scores = cross_validate(model, x_train, y_train, cv=5, scoring='r2',return_train_score=True)
        Test_r2 = scores['test_score']
        Train_r2 = scores['train_score']
        
        #MSE = cross_validation.cross_val_score(model, x_train, y_train, cv=5, scoring='mean_squared_error').mean()
            
        df = df.append({'Model': name, 'Aver. Test R2': Test_r2.mean(), 'Aver. Train R2': Train_r2.mean()}, ignore_index = True)
        
    return df.sort_values(by=['Aver. Test R2'], ascending=False)