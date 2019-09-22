# -*- coding: utf-8 -*-
import numpy as np

def normalize(X):
    mean = np.mean(X)
    std = np.std(x)
    
    return (X-mean) / std

def MSE(y, X, w):
    N = X.shape[0]
    
    return 1/(2*N) * (y - X@w).T @ (y - X@w)

def MSE_dw(y, X, w):
    N = X.shape[0]
    
    return -1/N * X.T @ (y - X@w)
    

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