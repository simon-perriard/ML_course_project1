import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from validation_helpers import *
from plots import *
from implementations import *



#import data
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


DATA_TEST_PATH = '../data/test.csv' 
y_donotUse, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


#split training data according to PRI_jet_num
split_x, split_y, split_ids = separate(y, tX, ids)

#compute percentage of null values in each columns
selection = dataStatistics(split_x)

#remove columns that only contains null values
cleaned = removeNone(split_x, selection)

#replace remaining null values with the column's median
cleaned_with_median = putMedianInsteadOfNone(cleaned)


#separate data with respect to column 24 and remove None
split_x_test, _, split_ids_test =  separate(y_donotUse, tX_test, ids_test)


split_x_cleaned_test = removeNone(split_x_test, dataStatistics(split_x_test))

#median instead of None
split_x_with_median = putMedianInsteadOfNone(split_x_cleaned_test)

split_x_with_median_with_momentum = add_momentum_vector(split_x_with_median)



#degrees for polynomial feature expension
degrees = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

y_res = []

acc = []

plot_data_per_jetnum = []


for i in range(len(cleaned_with_median)):
    
    
    
    #training: choose algorithm in the crossValidation function
    w_star, d, accuracy, training_set, plot_data = crossValidation(cleaned_with_median[i], split_y[i], 0.98, degrees ,6)
    
    
    #polynomial feature expension and normalization using the training data
    mean = np.mean(build_poly(training_set,d), axis = 0)
    std = np.std(build_poly(training_set,d), axis = 0)
    
      
    #put 1 if std = 0 (divide by 0 errors)
    std = std + (std == 0)
    
    extended_and_normalized = (build_poly(split_x_with_median[i], d) - mean) / std
    
    #adding bias term
    bias = np.ones(shape=split_x_with_median[i].shape)          
    x_test_ready = np.c_[bias, extended_and_normalized]
    
    #predictions
    y_res.append(predict_labels(w_star, x_test_ready))
    

    acc.append(accuracy)
    plot_data_per_jetnum.append(plot_data)

print("Accuracy per jet nbr: \n")
print(acc)


OUTPUT_PATH = '../data/submission.csv'
#reassemble the data for the submission
y_pred = put_together(y_res, split_ids_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)






