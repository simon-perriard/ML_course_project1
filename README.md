# ML_course_project1
ML class project that aims to identify the Higgs boson from original data from CERN, using Python ML libraries

# How to run the prediction

just execute in a shell : python run.py
The results will be available in the data folder


## Paramameters and accuracy obtained for each model

logistic 100 iter cross validation with accuracy  1 -16 polynomial degree lambda 0 , 0.3 in step of 0.1 training data 0.9:  77.6 %

logistic 100 iter cross validation with loss  1 -16 polynomial degree lambda 0 , 0.3 in step of 0.1 training data 0.9: 74.9 %

logistic regularized with 100 iter with loss 1-16 polynomial degree params lambdas = np.arange(0.000001,0.003,0.001)  gammas = np.arange(0.01,0.9,0.2), training data 0.9 : 75.9 %

logistic reg 100 iter cross validation with accuracy; gamma 0.01,0.9, step 0.2; lambda 0.0001, 0.3, step 0.1; splitRatio = 0.9 : 77.6 %

least squares with accuracy=>  with 98\% training data and degrees 1- 16 polynomial  : 82.8 % 

ridge regression with 98\% training data and degrees 1- 16 polynomial lambda 0.0001 to 0.001 by step of 0.0001: 80.5 %

least squares with loss 98% training set degree 1-16 poly : 82.8 %
