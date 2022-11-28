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

def run_models_multi_score(x_train: pd.DataFrame , y_train: pd.DataFrame):
    
    df = pd.DataFrame(columns=['Model']) 
    models = [
            ('LinReg', LinearRegression()), 
            ('RF', RandomForestRegressor()),
            ('XGB', XGBRegressor()),
            ('SVR',SVR(kernel="rbf", C=10, gamma="auto")),
            ('GBC',GradientBoostingRegressor())
    ]
        
    score = ["r2","neg_mean_squared_error","neg_root_mean_squared_error"]
    
    for name, model in models:
        
        scores = cross_validate(model, x_train, y_train, cv=5, scoring=score, return_train_score=True)
        
        df = df.append({'Model': name, 'Aver. Test R2': scores['test_r2'].mean(), 
                        'Aver. Train R2': scores['train_r2'].mean(),
                        'Aver. Test nMSE': scores['test_neg_mean_squared_error'].mean(), 
                        'Aver. Train nMSE': scores['train_neg_mean_squared_error'].mean(),
                        'Aver. Test nRMSE': scores['test_neg_root_mean_squared_error'].mean(),
                        'Aver. Train nRMSE': scores['train_neg_root_mean_squared_error'].mean()}, 
                        ignore_index = True)
        
    return df.sort_values(by=['Aver. Test R2'], ascending=False)
