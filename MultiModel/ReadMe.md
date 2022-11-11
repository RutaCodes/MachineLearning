*run_models_x.py* - function that uses cross validation to split specified dataset and compares results from multiple models 

This function uses default 5-fold cross validation data splitting strategy to compare scores (r2) from 5 models:
- Linear Regression 
- Random Forest Regressor
- Support Vector Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

A few algorithms used in this function have certain requirements to be used efficiently:
- SVM algorithms are not scale invariant, so it is highly recommended to scale data before using it to train model
- Regression models are sensitive to multicollinearity. It creates a problem in regression models because the inputs are all influencing each other, and it is difficult to efficeintly estimate paramater coeffieints.

Therefore, before using this function, data needs to be normalized/standardized/transformed (depends on original data distribution) and multicollinearity analysis needs to be performed. 
