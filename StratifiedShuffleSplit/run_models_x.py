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

def run_models_x(x_train: pd.DataFrame , y_train: pd.DataFrame):
    
    df = pd.DataFrame(columns=['Model']) 
    models = [
            ('LinReg', LinearRegression()), 
            ('RF', RandomForestRegressor()),
            ('XGB', XGBRegressor()),
            ('SVR',SVR(kernel="rbf", C=10, gamma="auto")),
            ('GBC',GradientBoostingRegressor()),
    ]
        
    
    for name, model in models:
        
        scores = cross_validate(model, x_train, y_train, cv=5, scoring='r2',return_train_score=True)
        Test_r2 = scores['test_score']
        Train_r2 = scores['train_score']
        
        df = df.append({'Model': name, 'Aver. Test R2': Test_r2.mean(), 'Aver. Train R2': Train_r2.mean()}, ignore_index = True)
        
    return df.sort_values(by=['Aver. Test R2'], ascending=False)
