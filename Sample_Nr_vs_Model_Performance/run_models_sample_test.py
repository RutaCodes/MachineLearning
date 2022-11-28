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
from sklearn.model_selection import cross_validate

def run_models_sample_test(x_train: pd.DataFrame , y_train: pd.DataFrame, score):
    
    models = [
            ('LinReg', LinearRegression()), 
            ('SVR',SVR(kernel="rbf", C=10, gamma="auto")),
            ('RF', RandomForestRegressor()),
            ('GBC',GradientBoostingRegressor()),
            ('XGB', XGBRegressor())
    ]
    
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    
    for name, model in models:
        
        scores = cross_validate(model, x_train, y_train, cv=5, scoring=score, return_train_score=True)
        
        df_test.loc[0,name] = scores['test_score'].mean()
        df_train.loc[0,name] = scores['train_score'].mean()
        
    return df_train, df_test