*run_models_multi_score.py* - function that uses cross validation to split specified dataset and compares results from multiple algorithms

This function uses default 5-fold cross validation to compare performance scores from 5 algorithms:
- Linear Regression 
- Random Forest Regression 
- Support Vector Regression 
- Gradient Boosting Regression 
- XGBoost Regression 

Cross validation uses 3 metrics to evaluate model performance:
- R^2
- MSE
- RMSE 

Scores from train and test models are reported.

*run_models_multi_score.py* - function that uses cross validation to evaluate multiple algorithms using specified score metric (e.g. RMSE) and returns train + test model results from each fold

*run_models_cross_val.py* - function that uses nested cross validation (*RepeatedKFold*) repeating cross validation with k-folds *n_repeat* number of times. Function returns specified cross validation score for each model run, which is equal to *n_splits* * *n_repeat*

A few algorithms used here have certain requirements:
- SVM algorithms are not scale invariant, so it is highly recommended to scale data before using it to train model
- Regression models are sensitive to multicollinearity. Collinearity creates a problem in regression models because the inputs are all influencing each other, and it is difficult to efficiently estimate paramater coefficients.

Therefore, before using these functions, data needs to be normalized/standardized/transformed (depends on data distribution) and multicollinearity analysis needs to be performed. 
