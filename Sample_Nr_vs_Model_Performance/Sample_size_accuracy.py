#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:16:01 2022

@author: rutabasijokaite
"""
# HOW DOES NUMBER OF TRAINING SAMPLES USED AFFECT MODEL ACCURACY?

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

#%% SPLITTING DATA AND MODELING

#Running models with cross validation
from run_models_sample_test import run_models_sample_test
#Choosing scoring metric
score = 'r2'

#Test different train/test ratios 
Ratio = np.array(range(5,96,5))/100 
df_train = pd.DataFrame()
df_test = pd.DataFrame()
for i in Ratio:
    x_train_rn, x_test_rn, y_train_rn, y_test_rn = train_test_split(
    Expl_var_wJ, Depend_var, test_size=i, random_state=42)
    Results = run_models_sample_test(x_train_rn, y_train_rn,score)
    Res_df_line = pd.concat([pd.DataFrame([i], columns=["Sample_size"]), Results[0]], axis=1)
    df_train = df_train.append(Res_df_line) #train results
    Res_df_line_test = pd.concat([pd.DataFrame([i], columns=["Sample_size"]), Results[1]], axis=1)
    df_test = df_test.append(Res_df_line_test) #test results
    
#%%    
#Convert % of samples to sample number 
Samp_nrs_split = (Ratio * len(Atrazine_conc_ar)).astype(int) #for test
Samp_nrs_split_train = ((1-Ratio) * len(Atrazine_conc_ar)).astype(int)  #for test

# Flipping data frame, son that it would start with the lowest number of training samples
df_flip = df_train[::-1] 

#%% Plot how model results depend on number of samples used during model training

plt.plot(Samp_nrs_split_train, df_flip["LinReg"].astype(float), label="LinReg") 
plt.plot(Samp_nrs_split_train, df_flip["SVR"].astype(float), label="SVR") 
plt.plot(Samp_nrs_split_train, df_flip["RF"].astype(float), label="RF") 
plt.plot(Samp_nrs_split_train, df_flip["GBC"].astype(float), label="GBC")
plt.plot(Samp_nrs_split_train, df_flip["XGB"].astype(float), label="XGBoost")  
plt.xlabel("Number of training samples")
plt.ylabel("R2")
plt.title("How models results depend on number of training samples?")
plt.legend()

#%%
plt.plot(Samp_nrs_split, df_test["LinReg"].astype(float), label="LinReg") 
plt.plot(Samp_nrs_split, df_test["SVR"].astype(float), label="SVR") 
plt.plot(Samp_nrs_split, df_test["RF"].astype(float), label="RF") 
plt.plot(Samp_nrs_split, df_test["GBC"].astype(float), label="GBC")
plt.plot(Samp_nrs_split, df_test["XGB"].astype(float), label="XGBoost")  
plt.xlabel("Number of test samples")
plt.ylabel("R2")
plt.title("How models results depend on number of test samples?")
plt.legend()
#%% See how this changes if dataset without outliers were used

# LOAD DATASET WITHOUT OUTLIERS

#Sample data, statial and temporal variables - trasformed (normalizing data distribution)
All_tr_variables_no_out_DF = pd.read_csv('All_variables_wJday_without_outliers.csv')
All_tr_variables_no_out_arr = np.array(All_tr_variables_no_out_DF)

#Import non-transformed concentration values
Atrazine_conc_no_out_DF = pd.read_csv('Atrazine_conc_org_without_outliers.csv')
Atrazine_conc_no_out_ar = np.array(Atrazine_conc_no_out_DF) 

#Predictor and dependant variables
Expl_var_wJ_no_out = np.delete(All_tr_variables_no_out_arr,[0, 1, 2],1).astype(float)
Depend_var_no_out = All_tr_variables_no_out_arr[:, 2].reshape(-1,1)

#%%
df_no_out_train = pd.DataFrame()
df_no_out_test = pd.DataFrame()
for i in Ratio:
    x_train_rn_no_out, x_test_rn_no_out, y_train_rn_no_out, y_test_rn_no_out = train_test_split(
    Expl_var_wJ_no_out, Depend_var_no_out, test_size=i, random_state=42)
    Results_no_out = run_models_sample_test(x_train_rn_no_out, y_train_rn_no_out,score)
    Res_df_line_no_out_train = pd.concat([pd.DataFrame([i], columns=["Sample_size"]), Results_no_out[0]], axis=1)
    Res_df_line_no_out_test = pd.concat([pd.DataFrame([i], columns=["Sample_size"]), Results_no_out[1]], axis=1)
    df_no_out_train = df_no_out_train.append(Res_df_line_no_out_train)
    df_no_out_test = df_no_out_test.append(Res_df_line_no_out_test)
    
    
df_no_out_train_flip = df_no_out_train[::-1] 
plt.plot(Samp_nrs_split_train, df_no_out_train_flip["LinReg"].astype(float), label="LinReg") 
plt.plot(Samp_nrs_split_train, df_no_out_train_flip["SVR"].astype(float), label="SVR") 
plt.plot(Samp_nrs_split_train, df_no_out_train_flip["RF"].astype(float), label="RF") 
plt.plot(Samp_nrs_split_train, df_no_out_train_flip["GBC"].astype(float), label="GBC")
plt.plot(Samp_nrs_split_train, df_no_out_train_flip["XGB"].astype(float), label="XGBoost")  
plt.xlabel("Number of training samples")
plt.ylabel("R2")
plt.title("How models results depend on number of training samples?")
plt.legend()