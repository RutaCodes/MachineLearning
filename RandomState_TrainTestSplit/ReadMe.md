# Problems with *train_test_split()* function

The *train_test_split()* function provided by the sklearn Python package is widely used for dividing data into train and test datasets. However, potential risks of using this function are rarely explored. 

The *random_state* hyperparameter in the train_test_split() function controls data shuffling. When *random_state* = *None*, shuffling process is out of developer's control and data is split into train and test datasets differently each time. When *random_state* = 1 (or any other integer - this integer sets the seed for the random generator), we get the same train - test data split across different execusions, which is crucial in order to create reproducable model results. Due to this *random_state* influence, *train_test_split()* function is considered to pseudorandomly separate data into train and test datasets. It is pseudorandom because while *random_state* seed value can change, each value corresponds to the same data subdivision. 

When *train_test_split()* function is used, random integer value is assigned to *random_state* in order to easily reproduce model results. The problem is that model results will vary based on which integer value is assigned to *random_state*, as data will be shuffled into train and test datasets differently, and hence, a different set of data points will be used to train ML model with each varying *random_state*.  

Here, I explore how one of the *train_test_split()* function inputs - *random_state* - influences model results. I use 5 different ML algorithms to develop a model capable of predicting daily micro-pollutant concentrations and analyze how changing *random_state* value affects cross validation score. 
