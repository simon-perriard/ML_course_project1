# -*- coding: utf-8 -*-
import numpy as np

def normalize(tx):
    mean = np.mean(tx)
    std = np.std(tx)
    
    return (tx-mean) / std

def MSE(y, tx, w):
    e = y- tx@w
    
    return 1/2 * np.mean(e**2)

def MSE_dw(y, tx, w):
    e = y - tx@w
    return -tx.T@e/len(e)
    

def least_squares_GD(y, tx, initial_w, max_iter, gamma):
       
    next_w = initial_w
        
    for i in range(max_iter):
        current_w = next_w
        next_w = current_w - gamma * MSE_dw(y, tx, current_w)
            
    return (next_w, MSE(y, tx, next_w))
    
    
def least_squares_SGD(y, tx, initial_w, max_iter, gamma):
    
    next_w = initial_w
    
    mini_batch_size = 1

    random_indexes = np.zeros((1, mini_batch_size))
    
    for i in range(max_iter):
        
        random_indexes = np.random.uniform(0, tx.shape[0], mini_batch_size)
        
        mini_batch_X = tx[random_indexes]
        mini_batch_y = y[random_indexes]
        
        current_w = next_w
        next_w = current_w - gamma * MSE_dw(mini_batch_y, mini_batch_X, current_w)
    
    return (next_w, MSE(y, tx, new_w))