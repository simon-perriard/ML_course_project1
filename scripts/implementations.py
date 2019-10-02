# -*- coding: utf-8 -*-
import numpy as np

def normalize(tx):
    mean = np.mean(tx)
    std = np.std(tx)
    
    return (tx-mean) / std

def MSE_loss(y, tx, w):
    e = y - tx@w
    
    return 1/2 * np.mean(e**2)

def MSE_gradient(y, tx, w):
    e = y - tx@w
    return -tx.T@e/len(e)
    

def least_squares_GD(y, tx, initial_w, max_iter, gamma):
       
    w = initial_w
        
    for i in range(max_iter):
        w = w - gamma * MSE_gradient(y, tx, current_w)
            
    return (w, MSE_loss(y, tx, w))
    
    
def least_squares_SGD(y, tx, initial_w, max_iter, gamma):
    
    w = initial_w
    
    mini_batch_size = 1
    mini_batch_indices_to_take = np.arange(mini_batch_size)

    
    for i in range(max_iter):
        shuffled_indices = np.random.permutation(np.arange(len(w)))
        
        mini_batch_indices = np.take(shuffled_indices, mini_batch_indices_to_take)
        
        mini_batch_X = tx[mini_batch_indices]
        mini_batch_y = y[mini_batch_indices]
        
        w = w - gamma * MSE_gradient(mini_batch_y, mini_batch_X, w)
    
    return (w, MSE_loss(y, tx, w))