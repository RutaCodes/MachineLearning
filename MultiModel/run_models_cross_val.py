#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:26:35 2022

@author: rutabasijokaite
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import cross_validate, RepeatedKFold

def run_models_cross_val(x_train: pd.DataFrame , y_train: pd.DataFrame, score):
    
    models = [
            ('LinReg', LinearRegression()), 
            ('RF', RandomForestRegressor()),
            ('XGB', XGBRegressor()),
            ('SVR',SVR(kernel="rbf", C=10, gamma="auto")),
            ('GBC',GradientBoostingRegressor()),
    ]
    
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    
    for name, model in models:
        
        cv_c = RepeatedKFold(n_splits=5, n_repeats=5, random_state=23)
        scores = cross_validate(model, x_train, y_train, cv=cv_c, scoring=score, return_train_score=True)
        
        df_test.loc[:,name] = scores['test_score']
        df_train.loc[:,name] = scores['train_score']
        
    return df_test, df_train  