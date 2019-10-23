# -*- coding: utf-8 -*-
import numpy as np

def normalize(tx):
    mean = np.mean(tx, axis=0)
    std = np.std(tx, axis=0)
    
    for i in range(len(std)):
        
        if(std[i] == 0):
            std[i] = 1
        
    return (tx-mean) / std


def MSE_loss(y, tx, w):
    e = y - tx@w
    
    return np.mean(e**2)/2


def MSE_gradient(y, tx, w):
    
    e = y - tx@w
    return -tx.T@e/len(e)
    

def least_squares_GD(y, tx, initial_w, max_iter, gamma):
       
    w = initial_w
        
    for i in range(max_iter):
        w = w - gamma * MSE_gradient(y, tx, w)
            
    return (w, MSE_loss(y, tx, w))
    
    
def least_squares_SGD(y, tx, initial_w, max_iter, gamma):
    
    w = initial_w
    
    mini_batch_size = 1
    mini_batch_indices_to_take = np.arange(mini_batch_size)
    indices = np.arange(len(y))

    
    for i in range(max_iter):
        shuffled_indices = np.random.permutation(indices)
        
        mini_batch_indices = np.take(shuffled_indices, mini_batch_indices_to_take)
        
        mini_batch_X = tx[mini_batch_indices]
        mini_batch_y = y[mini_batch_indices]
        
        w = w - gamma * MSE_gradient(mini_batch_y, mini_batch_X, w)
    
    return (w, MSE_loss(y, tx, w))


def least_squares(y, tx):
    
    w = np.linalg.lstsq(tx.T@tx, tx.T@y, rcond=None)[0]
    
    return (w, MSE_loss(y, tx, w))


def ridge_regression(y, tx, lambda_):
    
   
    
    w = np.linalg.lstsq(tx.T@tx + 2 * len(y) * lambda_ * np.identity(tx.shape[1]), tx.T@y, rcond=None)[0]
    
    
    return (w, MSE_loss(y, tx, w) )


def logistic_loss(y, tx, w):
    
    pred = sigmoid(tx @ w)

    #print("prediction")
    #print(pred)
    #print(pred[np.where(pred != 0)])
    #loss = -y.T @ np.log(pred) - (1 - y).T @ np.log(1 - pred)  gives erros 
    #loss = np.log(1 + np.exp(pred)) - y.T@(pred) #give infinity
    #loss = np.sum(loss , axis = 0)
    loss = 0
    
    for n in range (len(y)):
        
        if(y[n] == 1):
            loss += np.log(pred[n])  
        else:
            loss += np.log(1 - pred[n])
            
    return  - loss


def sigmoid(z):
    
    return 1.0/(1 + np.exp(-z))


def logistic_gradient(y, tx, w):
    
    return tx.T@(sigmoid(tx@w) - y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):

    t = np.ones(len(y))
    t[np.where(y == -1)] = 0
    
    w = initial_w

    for i in range(max_iters):
        w = w - gamma * logistic_gradient(t, tx, w)
        
    
    return (w, logistic_loss(t, tx, w))




def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    t = np.ones(len(y))
    t[np.where(y == -1)] = 0
    
    w = initial_w

    for i in range(max_iters):
        w = w - gamma * (logistic_gradient(t, tx, w) + lambda_  * w)
    
    loss = logistic_loss(t, tx, w) + lambda_ * np.sum(w**2) 
    
    return (w, loss)











