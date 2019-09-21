# -*- coding: utf-8 -*-
import numpy as np

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