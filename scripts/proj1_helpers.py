# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
            
            
            
            
#method to split the training set into a (new) training set and a test set (same as in lab03)

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
 
    # split the data based on the given ratio

    training_nbr = int(x.shape[0] * ratio)
    indexes = np.random.choice(x.shape[0],training_nbr, replace=False)
    
    x_train = x[indexes]
    y_train = y[indexes]
    x_test = np.delete(x, indexes, axis = 0)
    y_test = np.delete(y, indexes, axis = 0)
    
    
    return x_train, y_train, x_test, y_test
    
    
    
 
 
 
#method to perform polynomial feature expension

def build_poly(x, degree):
   
    x_extended = x

    for d in range (2, degree +1):
        x_extended = np.c_[x_extended, x**d]
        

    return x_extended   
    
    
    

#Instead of putting the median we can simply drop the data where columns 0 == -999
def dropLineIfNone(cleaned, split_y, split_ids):
    
    res_x=[]
    res_y=[]
    res_ids=[]
    
    for i in range(len(cleaned)):
        
        current = cleaned[i]
        
        drop_indexes = np.where(current[:,0] != -999)
        
        res_x.append(current[drop_indexes])
        res_y.append(current[drop_indexes])
        res_ids.append(current[drop_indexes])
        
    return res_x, res_y, res_ids   
    
    
    
    
#replace the value of column 0 (can be None sometimes) by the median value of this column

def putMedianInsteadOfNone(cleaned):
    
    completed_data = []
    
    for i in range(len(cleaned)):
        #current PRI_jet_num
        current = cleaned[i]
        
        median = np.median(current[np.where(current[:,0] != -999)], axis = 0)
        
        #replace -999 by median value
        current[np.where(current[:,0] == -999)] = median
        
        completed_data.append(current)
    
    
    return completed_data
    
    
    
    
#add columns by computing different values that were mentionned in the cern pdf
    
def add_momentum_vector(data):
#### colums with tau info ####
    tau_transverse_momentum_0_1_col = 9
    tau_pseudo_rapidity_0_1_col = 10
    tau_azimuth_angle_0_1_col = 11
    
    tau_transverse_momentum_2_3_col = 13
    tau_pseudo_rapidity_2_3_col = 14
    tau_azimuth_angle_2_3_col = 15
    
#### colums with lep info ####    
    lep_transverse_momentum_0_1_col = 12
    lep_pseudo_rapidity_0_1_col = 13
    lep_azimuth_angle_0_1_col = 14
     
    lep_transverse_momentum_2_3_col = 16
    lep_pseudo_rapidity_2_3_col = 17
    lep_azimuth_angle_2_3_col = 18

#### colums with leading_jet info ####
    leading_jet_transverse_momentum_1_col = 19
    leading_jet_pseudo_rapidity_1_col = 20
    leading_jet_azimuth_angle_1_col = 21

    leading_jet_transverse_momentum_2_3_col = 23
    leading_jet_pseudo_rapidity_2_3_col = 24
    leading_jet_azimuth_angle_2_3_col = 25
    
#### colums with subleading_jet info ####
    subleading_jet_transverse_momentum_2_3_col = 26
    subleading_jet_pseudo_rapidity_2_3_col = 27
    subleading_jet_azimuth_angle_2_3_col = 28

#### columns with missing transverse energy info ####
    mte_0_1_col = 15
    mte_angle_0_1_col = 16
    
    mte_2_3_col = 19
    mte_angle_2_3_col = 20
    
    data_with_momentum_vector = []
    
    
    for i in range(len(data)):
        current = data[i]
        
        if i == 0:
            tau_transverse_momentum_0 = current[:,tau_transverse_momentum_0_1_col]
            tau_pseudo_rapidity_0 = current[:, tau_pseudo_rapidity_0_1_col]
            tau_azimuth_angle_0 = current[:, tau_azimuth_angle_0_1_col]
            
            lep_transverse_momentum_0 = current[:,lep_transverse_momentum_0_1_col]
            lep_pseudo_rapidity_0 = current[:, lep_pseudo_rapidity_0_1_col]
            lep_azimuth_angle_0 = current[:, lep_azimuth_angle_0_1_col]
            
            mte_0 = current[:, mte_0_1_col]
            mte_angle_0 = current[:, mte_angle_0_1_col]
            
            current = np.c_[current, tau_transverse_momentum_0*np.cos(tau_azimuth_angle_0)]
            current = np.c_[current, tau_transverse_momentum_0*np.sin(tau_azimuth_angle_0)]
            current = np.c_[current, tau_transverse_momentum_0*np.sinh(tau_pseudo_rapidity_0)]
            current = np.c_[current, tau_transverse_momentum_0*np.cosh(tau_pseudo_rapidity_0)]
            
            current = np.c_[current, lep_transverse_momentum_0*np.cos(lep_azimuth_angle_0)]
            current = np.c_[current, lep_transverse_momentum_0*np.sin(lep_azimuth_angle_0)]
            current = np.c_[current, lep_transverse_momentum_0*np.sinh(lep_pseudo_rapidity_0)]
            current = np.c_[current, lep_transverse_momentum_0*np.cosh(lep_pseudo_rapidity_0)]
            
            current = np.c_[current, mte_0*np.cos(mte_angle_0)]
            current = np.c_[current, mte_0*np.sin(mte_angle_0)]
            
        elif i == 1:
            tau_transverse_momentum_1 = current[:,tau_transverse_momentum_0_1_col]
            tau_pseudo_rapidity_1 = current[:, tau_pseudo_rapidity_0_1_col]
            tau_azimuth_angle_1 = current[:, tau_azimuth_angle_0_1_col]
            
            lep_transverse_momentum_1 = current[:,lep_transverse_momentum_0_1_col]
            lep_pseudo_rapidity_1 = current[:, lep_pseudo_rapidity_0_1_col]
            lep_azimuth_angle_1 = current[:, lep_azimuth_angle_0_1_col]
            
            leading_jet_transverse_momentum_1 = current[:,lep_transverse_momentum_0_1_col]
            leading_jet_pseudo_rapidity_1 = current[:, lep_pseudo_rapidity_0_1_col]
            leading_jet_azimuth_angle_1 = current[:, lep_azimuth_angle_0_1_col]
            
            mte_1 = current[:, mte_0_1_col]
            mte_angle_1 = current[:, mte_angle_0_1_col]
            
            current = np.c_[current, tau_transverse_momentum_1*np.cos(tau_azimuth_angle_1)]
            current = np.c_[current, tau_transverse_momentum_1*np.sin(tau_azimuth_angle_1)]
            current = np.c_[current, tau_transverse_momentum_1*np.sinh(tau_pseudo_rapidity_1)]
            current = np.c_[current, tau_transverse_momentum_1*np.cosh(tau_pseudo_rapidity_1)]
            
            current = np.c_[current, lep_transverse_momentum_1*np.cos(lep_azimuth_angle_1)]
            current = np.c_[current, lep_transverse_momentum_1*np.sin(lep_azimuth_angle_1)]
            current = np.c_[current, lep_transverse_momentum_1*np.sinh(lep_pseudo_rapidity_1)]
            current = np.c_[current, lep_transverse_momentum_1*np.cosh(lep_pseudo_rapidity_1)]
            
            current = np.c_[current, leading_jet_transverse_momentum_1*np.cos(leading_jet_azimuth_angle_1)]
            current = np.c_[current, leading_jet_transverse_momentum_1*np.sin(leading_jet_azimuth_angle_1)]
            current = np.c_[current, leading_jet_transverse_momentum_1*np.sinh(leading_jet_pseudo_rapidity_1)]
            current = np.c_[current, leading_jet_transverse_momentum_1*np.cosh(leading_jet_pseudo_rapidity_1)]
            
            current = np.c_[current, mte_1*np.cos(mte_angle_1)]
            current = np.c_[current, mte_1*np.sin(mte_angle_1)]
            
        else:
            tau_transverse_momentum_2_3 = current[:,tau_transverse_momentum_2_3_col]
            tau_pseudo_rapidity_2_3 = current[:, tau_pseudo_rapidity_2_3_col]
            tau_azimuth_angle_2_3 = current[:, tau_azimuth_angle_2_3_col]
            
            lep_transverse_momentum_2_3 = current[:,lep_transverse_momentum_2_3_col]
            lep_pseudo_rapidity_2_3 = current[:, lep_pseudo_rapidity_2_3_col]
            lep_azimuth_angle_2_3 = current[:, lep_azimuth_angle_2_3_col]
            
            leading_jet_transverse_momentum_2_3 = current[:,leading_jet_transverse_momentum_2_3_col]
            leading_jet_pseudo_rapidity_2_3 = current[:, leading_jet_pseudo_rapidity_2_3_col]
            leading_jet_azimuth_angle_2_3 = current[:, leading_jet_azimuth_angle_2_3_col]
            
            subleading_jet_transverse_momentum_2_3 = current[:,subleading_jet_transverse_momentum_2_3_col]
            subleading_jet_pseudo_rapidity_2_3 = current[:, subleading_jet_pseudo_rapidity_2_3_col]
            subleading_jet_azimuth_angle_2_3 = current[:, subleading_jet_azimuth_angle_2_3_col]
            
            mte_2_3 = current[:, mte_2_3_col]
            mte_angle_2_3 = current[:, mte_angle_2_3_col]
            
            current = np.c_[current, tau_transverse_momentum_2_3*np.cos(tau_azimuth_angle_2_3)]
            current = np.c_[current, tau_transverse_momentum_2_3*np.sin(tau_azimuth_angle_2_3)]
            current = np.c_[current, tau_transverse_momentum_2_3*np.sinh(tau_pseudo_rapidity_2_3)]
            current = np.c_[current, tau_transverse_momentum_2_3*np.cosh(tau_pseudo_rapidity_2_3)]
            
            current = np.c_[current, lep_transverse_momentum_2_3*np.cos(lep_azimuth_angle_2_3)]
            current = np.c_[current, lep_transverse_momentum_2_3*np.sin(lep_azimuth_angle_2_3)]
            current = np.c_[current, lep_transverse_momentum_2_3*np.sinh(lep_pseudo_rapidity_2_3)]
            current = np.c_[current, lep_transverse_momentum_2_3*np.cosh(lep_pseudo_rapidity_2_3)]
            
            current = np.c_[current, leading_jet_transverse_momentum_2_3*np.cos(leading_jet_azimuth_angle_2_3)]
            current = np.c_[current, leading_jet_transverse_momentum_2_3*np.sin(leading_jet_azimuth_angle_2_3)]
            current = np.c_[current, leading_jet_transverse_momentum_2_3*np.sinh(leading_jet_pseudo_rapidity_2_3)]
            current = np.c_[current, leading_jet_transverse_momentum_2_3*np.cosh(leading_jet_pseudo_rapidity_2_3)]
        
            current = np.c_[current, subleading_jet_transverse_momentum_2_3*np.cos(subleading_jet_azimuth_angle_2_3)]
            current = np.c_[current, subleading_jet_transverse_momentum_2_3*np.sin(subleading_jet_azimuth_angle_2_3)]
            current = np.c_[current, subleading_jet_transverse_momentum_2_3*np.sinh(subleading_jet_pseudo_rapidity_2_3)]
            current = np.c_[current, subleading_jet_transverse_momentum_2_3*np.cosh(subleading_jet_pseudo_rapidity_2_3)]
        
            current = np.c_[current, mte_2_3*np.cos(mte_angle_2_3)]
            current = np.c_[current, mte_2_3*np.sin(mte_angle_2_3)]
                    
        data_with_momentum_vector.append(current)
    
    return data_with_momentum_vector
    
    
    
    
    
    
    
    
#Since we separated the data according to PRI_jet_num
# we have to make separate prediction and then put them together for the submission

def put_together(labels, indices):
    
    #First build first chunk
    ids_0 = np.matrix(indices[0]).T
    lab_0 = np.matrix(labels[0]).T
    
    unsorted_res = np.concatenate((ids_0, lab_0), axis=1)
    
    for i in range(1,len(labels)):
        ids = np.matrix(indices[i]).T
        lab = np.matrix(labels[i]).T
        by_jet_num = np.concatenate((ids, lab), axis=1)
        unsorted_res = np.concatenate((unsorted_res, by_jet_num), axis=0)
    
    sorted_res = unsorted_res[np.lexsort(np.fliplr(unsorted_res).T)]
    
    return sorted_res[0,:,:][:,1]
    
    
    
    
    
    
    
#print statistics about the None values (-999) for each columns
#returns a boolean array that can be used to filter the columns that have 100% of undefined values (-999)
def dataStatistics(data):
    
    stats=[]
    
    for i in range(len(data)):
        
        print("Statistics ")
        print("Type :")
        print(i)
        
        
        nones = (data[i] == -999)
    
        mean = np.sum(nones, axis=0)/nones.shape[0]
        print(mean) 
        stats.append(mean != 1)
    
    return stats
    
    
    
    
#remove the columns from each set of data given a boolean array

def removeNone(data, selection):
   
    cleaned=[]
    
    for i in range(4):
        curr_data = data[i]
        
        cleaned.append(curr_data[:,selection[i]])
      
    return cleaned
    
    
    
    
# Separate the data according to the value of column 24 (PRI_jet_num) 

def separate(y, tX, ids):
    
    split_x = []
    split_y = []
    split_ids = []
    
    jet_column_nbr = 22
    
    for i in range(4):
        
        split_x.append(tX[np.where(tX[:,jet_column_nbr] == i)])
        split_y.append(y[np.where(tX[:,jet_column_nbr] == i)])
        split_ids.append(ids[np.where(tX[:,jet_column_nbr] == i)])
    
    
    
    return split_x, split_y, split_ids
    
