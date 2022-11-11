*SVR_hyper.py* - function that uses GridSearchCV to find optimal Support Vector Regressor model hyper-parameters for a given dataset

# Tuning the hyper-parameters of an estimator

As hyper-parameters are not directly learnt within estimators, it is recommended to search the hyper-parameter space for the best cross validation score. Any parameter can be optimized by cross-validated grid search over a parameter grid. Two generic approaches exist in *scikit-learn*:
- *GridSearchCV* exhaustively considers all parameter combinations from a specified grid of parameter values and the best combination is retained
- *RandomizedSearchCV* samples a given number of candidates from a parameter space with a specified distribution 

Note that it is common that a subset of those parameters can have a large impact on model results, while others might be left to their default values.

More about hyper-parameter tuning using *scikit-learn*:
https://scikit-learn.org/stable/modules/grid_search.html

More about GridSearchCV:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
