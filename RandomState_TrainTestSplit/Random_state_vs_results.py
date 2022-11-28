#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:34:28 2022

@author: rutabasijokaite
"""

# HOW DOES RANDOM STATE IN TRAIN/TEST SPLIT FUNCTION AFFECT MODEL ACCURACY?

#%% IMPORTING LIBRARIES

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%% LOADING DATA - test data with outliers

#Sample data, statial and temporal variables - trasformed (normalizing data distribution)
All_tr_variables_DF = pd.read_csv('All_transf_sampl_variables_1114.csv')
All_tr_variables_arr = np.array(All_tr_variables_DF)

#Import non-transformed concentration values
Atrazine_conc_DF = pd.read_csv('Atraz_samp_final_list_1031.csv')
Atrazine_conc_ar = np.array(Atrazine_conc_DF)

#Defining predictor and dependant variables
Expl_var_wJ = np.delete(All_tr_variables_arr,[0, 1, 2],1).astype(float)
Depend_var = All_tr_variables_arr[:, 2].reshape(-1,1)

#%% TESTING RANDOM STATES

Rand_state = np.array(range(0,101))

#Running models with cross validation
from run_models_sample_test import run_models_sample_test
#Choosing scoring metric
score = 'r2'

df_train_rs = pd.DataFrame()
df_test_rs = pd.DataFrame()
for i in Rand_state:
    x_train_rs, x_test_rs, y_train_rs, y_test_rs = train_test_split(
    Expl_var_wJ, Depend_var, test_size=0.2, random_state=i)
    Results = run_models_sample_test(x_train_rs, y_train_rs, score)
    Res_df_line_rs = pd.concat([pd.DataFrame([i], columns=["Random_state"]), Results[0]], axis=1)
    df_train_rs = df_train_rs.append(Res_df_line_rs) #train results
    Res_df_line_rs_test = pd.concat([pd.DataFrame([i], columns=["Random_state"]), Results[1]], axis=1)
    df_test_rs = df_test_rs.append(Res_df_line_rs_test) #test results
    
#%%
# Plotting results
plt.plot(df_train_rs["Random_state"].astype(float), df_train_rs["LinReg"].astype(float), label="LinReg") 
plt.plot(df_train_rs["Random_state"].astype(float), df_train_rs["SVR"].astype(float), label="SVR") 
plt.plot(df_train_rs["Random_state"].astype(float), df_train_rs["RF"].astype(float), label="RF") 
plt.plot(df_train_rs["Random_state"].astype(float), df_train_rs["GBC"].astype(float), label="GBC")
plt.plot(df_train_rs["Random_state"].astype(float), df_train_rs["XGB"].astype(float), label="XGBoost")  
plt.xlabel("Random State")
plt.ylabel("R2")
plt.title("R2 vs Random State (Train data)")
plt.legend()
plt.savefig("Random_state_results_train.pdf")
#%%
# Plotting results
plt.plot(df_test_rs["Random_state"].astype(float), df_test_rs["LinReg"].astype(float), label="LinReg") 
plt.plot(df_test_rs["Random_state"].astype(float), df_test_rs["SVR"].astype(float), label="SVR") 
plt.plot(df_test_rs["Random_state"].astype(float), df_test_rs["RF"].astype(float), label="RF") 
plt.plot(df_test_rs["Random_state"].astype(float), df_test_rs["GBC"].astype(float), label="GBC")
plt.plot(df_test_rs["Random_state"].astype(float), df_test_rs["XGB"].astype(float), label="XGBoost")  
plt.xlabel("Random State")
plt.ylabel("R2")
plt.title("R2 vs Random State (Test data)")
plt.legend()
plt.savefig("Random_state_results_test.pdf")
#%% Which random state has the highest cumulative R2?
Cum_r2 =  np.sum(np.array(df_test_rs.iloc[:,range(1,6)]), axis=1)
np.where(Cum_r2 == np.max(Cum_r2))[0]
#Random state = 23 leads to highest culumative r2 value for all models

#Test if its is the same for top 3 models: RF, GBC, XGBoost
Cum_r2_h =  np.sum(np.array(df_test_rs.iloc[:,range(3,6)]), axis=1)
np.where(Cum_r2_h == np.max(Cum_r2_h))[0]
# It is, since r2 are the highest for these 3 models therefore they influnce cumulative r2 the most