*run_models_multi_score.py* - function that uses cross validation to split specified dataset and compares results from multiple models 

This function uses default 5-fold cross validation data splitting strategy to compare scores (e.g. RMSE) from 5 models:
- Linear Regression 
- Random Forest Regressor
- Support Vector Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Cross validation uses 3 scores to evaluate model performance:
- R2
- MSE
- RMSE
Scores from train and test sets are reported.

*run_models_multi_score.py* - function that uses cross validation to evaluate multiple models using specified score metric (e.g. R2) and returns train + test results from each fold

A few algorithms used in this function have certain requirements to be used efficiently:
- SVM algorithms are not scale invariant, so it is highly recommended to scale data before using it to train model
- Regression models are sensitive to multicollinearity. It creates a problem in regression models because the inputs are all influencing each other, and it is difficult to efficeintly estimate paramater coeffieints.

Therefore, before using this function, data needs to be normalized/standardized/transformed (depends on original data distribution) and multicollinearity analysis needs to be performed. 
