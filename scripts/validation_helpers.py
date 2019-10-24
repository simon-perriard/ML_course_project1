# -*- coding: utf-8 -*-
import numpy as np
from proj1_helpers import *
from implementations import *

#perform cross-validation using the accuracy 

def crossValidation(x, y, splitRatio, degrees, seed =1):
    
    x_train, y_train, x_test, y_test = split_data(x, y, splitRatio, seed)
    
    a_training = []
    a_testing = []
    weights = []
    degr = []
    
    plot_data = []

    
    # define parameter (just add more for loops if there are more parameters for the model)
    #lambdas = np.arange(0.000001,0.00001,0.000001)
    lambdas = [0]
    #lambdas = np.arange(0,0.3,0.1)
    
    for ind, lambda_ in enumerate(lambdas):
        
        for ind_d, d in enumerate(degrees):
            
            
            #perform polynomial feature expension
            x_test_poly = build_poly(x_test,d)
            x_train_poly = build_poly(x_train, d)
           
            
            #normalize data (DANGER: the test set must be normalized with the training set's mean and std)
            mean = np.mean(x_train_poly, axis =0)
            std = np.std(x_train_poly, axis = 0)
            
              
            #put 1 if std = 0
            std = std + (std == 0)

            
            x_train_ready = (x_train_poly - mean) / std
            x_test_ready = (x_test_poly - mean) / std
            
            
            #add bias term
            bias_tr = np.ones(shape=x_train.shape)
            bias_te = np.ones(shape=x_test.shape)
            
            x_train_ready = np.c_[bias_tr, x_train_ready]
            x_test_ready = np.c_[bias_te, x_test_ready]
            
            
            #Models
        
            #w_star, e_tr = ridge_regression(y_train,x_train_ready, lambda_)
        
            #w_star, e_tr = logistic_regression(y_train, x_train_ready,np.ones(x_train_ready.shape[1])  ,100, lambda_)
        
            #don't usel least squares with lambda bigger than 0.35 ideal: lambdas = np.arange(0.001,0.13,0.01)
            #w_star, e_tr = least_squares_GD(y_train, x_train_ready,np.ones(x_train_ready.shape[1])  ,400, lambda_)    
            #w_star, e_tr = least_squares_SGD(y_train, x_train,np.ones(x_train.shape[1])  ,400, lambda_)
        
            #closed-form least squares
            w_star, e_tr = least_squares(y_train, x_train_ready)  
        
            degr.append(d)
        
            #compare the prediction with the reality
            accuracy_training = np.count_nonzero(predict_labels(w_star, x_train_ready) + y_train)/len(y_train)
            accuracy_testing = np.count_nonzero(predict_labels(w_star, x_test_ready) + y_test)/len(y_test)
            
            #compare the prediction with the reality for logistic regression
            #accuracy_training = np.count_nonzero(predict_labels_logistic(w_star, x_train_ready) + y_train)/len(y_train)
            #accuracy_testing = np.count_nonzero(predict_labels_logistic(w_star, x_test_ready) + y_test)/len(y_test)
        
            a_training.append(accuracy_training)
            a_testing.append(accuracy_testing)
            weights.append(w_star)
            plot_data.append((lambda_, d, accuracy_testing))
            print("lambda={l:.5f},degree={deg}, Training Accuracy={tr}, Testing Accuracy={te}".format(
                   l=lambda_, tr=a_training[ind*len(degrees)+ind_d], te=a_testing[ind*len(degrees)+ind_d], deg=d))
        
            
    
    return weights[np.argmax(a_testing)], degr[np.argmax(a_testing)], a_testing[np.argmax(a_testing)], x_train, plot_data
    
    
    
    
    
#perform cross-validation using the loss instead of the accuracy 

def crossValidation_with_loss(x, y, splitRatio, degrees, seed =1):
    
    x_train, y_train, x_test, y_test = split_data(x, y, splitRatio, seed)
    
    loss_tr = []
    loss_te = []
    weights = []
    degr = []
    
    
    plot_data = []
    
    # define parameter (just add more for loops if there are more parameters for the model)
    #lambdas = np.arange(0.000001,0.00001,0.000001)
    lambdas = np.arange(0,0.3,0.1)
    
    for ind, lambda_ in enumerate(lambdas):
        
        for ind_d, d in enumerate(degrees):
            
            
            #perform polynomial feature expension
            x_test_poly = build_poly(x_test,d)
            x_train_poly = build_poly(x_train, d)
           
            
            #normalize data (DANGER: the test set must be normalized with the training set's mean and std)
            mean = np.mean(x_train_poly, axis =0)
            std = np.std(x_train_poly, axis = 0)
            
              
            #put 1 if std = 0
            std = std + (std == 0)

            
            x_train_ready = (x_train_poly - mean) / std
            x_test_ready = (x_test_poly - mean) / std
            
            
            #add bias term
            bias_tr = np.ones(shape=x_train.shape)
            bias_te = np.ones(shape=x_test.shape)
            
            x_train_ready = np.c_[bias_tr, x_train_ready]
            x_test_ready = np.c_[bias_te, x_test_ready]
            
            
            #Models
        
            #w_star, e_tr = ridge_regression(y_train,x_train_ready, lambda_)
                    
            w_star, e_tr = logistic_regression(y_train, x_train_ready,np.ones(x_train_ready.shape[1])  ,400, lambda_)
        
            #don't usel least squares with lambda bigger than 0.35 ideal: lambdas = np.arange(0.001,0.13,0.01)
            #w_star, e_tr = least_squares_GD(y_train, x_train_ready,np.ones(x_train_ready.shape[1])  ,400, lambda_)    
            #w_star, e_tr = least_squares_SGD(y_train, x_train,np.ones(x_train.shape[1])  ,400, lambda_)
        
            #closed-form least squares
            #w_star, e_tr = least_squares(y_train, x_train_ready)  
        
            degr.append(d)
        
        
            #compute the loss on the test set
            #loss for least squares
            #e_te = MSE_loss(y_test, x_test_ready, w_star)
            
            
            #loss for logistic regression
            
            #need to map the y= -1 => y = 0  for logistic regression 's loss computation
            t = np.ones(len(y_test))
            t[np.where(y_test == -1)] = 0
            e_te = logistic_loss(t, x_test_ready, w_star)
            
            
            plot_data.append((lambda_, d, e_te))
        
            loss_tr.append(e_tr)
            loss_te.append(e_te)
            weights.append(w_star)
            print("lambda={l:.5f},degree={deg}, Training Loss={tr}, Testing Loss={te}".format(
                   l=lambda_, tr=loss_tr[ind*len(degrees)+ind_d], te=loss_te[ind*len(degrees)+ind_d], deg=d))
        
            
    
    return weights[np.argmin(loss_te)], degr[np.argmin(loss_te)], loss_te[np.argmin(loss_te)], x_train, plot_data
    
    
    

#perform cross-validation 

def crossValidationForLogistic_reg(x, y, splitRatio, degrees, seed =1):
    
    x_train, y_train, x_test, y_test = split_data(x, y, splitRatio, seed)
    
    a_training = []
    a_testing = []
    weights = []
    degr = []
    
    index = 0
    plot_data = []
    
    # define parameter (just add more for loops if there are more parameters for the model)
    lambdas = np.arange(0.0001,0.3,0.1)
    gammas = np.arange(0.01,0.9,0.2)
    
    for ind, lambda_ in enumerate(lambdas):
        
        for ind_d, d in enumerate(degrees):
            
            for ind_g, gamma in enumerate(gammas):
            
                #perform polynomial feature expension
                x_test_poly = build_poly(x_test,d)
                x_train_poly = build_poly(x_train, d)
            
                #normalize data (DANGER: the test set must be normalized with the training set's mean and std)
                mean = np.mean(x_train_poly, axis =0)
                std = np.std(x_train_poly, axis = 0)
            
                #put 1 if std = 0
                std = std + (std == 0)
            
                x_train_ready = (x_train_poly - mean) / std
                x_test_ready = (x_test_poly - mean) / std
                
               
                #add bias term
                
                bias_tr = np.ones(shape=x_train.shape)
                bias_te = np.ones(shape=x_test.shape)
            
                x_train_ready = np.c_[bias_tr, x_train_ready]
                x_test_ready = np.c_[bias_te, x_test_ready]
                
           

                #Model

                w_star, e_tr = reg_logistic_regression(y_train, x_train_ready, lambda_, np.ones(x_test_ready.shape[1]), 30, gamma)
        
           
                degr.append(d)
        
                #compare the prediction with the reality
                accuracy_training = np.count_nonzero(predict_labels_logistic(w_star, x_train_ready) + y_train)/len(y_train)
                accuracy_testing = np.count_nonzero(predict_labels_logistic(w_star, x_test_ready) + y_test)/len(y_test)
                
                plot_data.append((lambda_, d, gamma, accuracy_testing))

                a_training.append(accuracy_training)
                a_testing.append(accuracy_testing)
                weights.append(w_star)
                print("lambda={l:.5f},degree={deg}, gamma={ga:.5f}, Training Accuracy={tr}, Testing Accuracy={te}".format(
                       l=lambda_, tr=a_training[index], te=a_testing[index], deg=d, ga=gamma))
        
                #increment index
                index = index + 1
    
    return weights[np.argmax(a_testing)], degr[np.argmax(a_testing)], a_testing[np.argmax(a_testing)], x_train, plot_data  
    
    
    
    
    
    
    
    
    
    

#perform cross-validation using the loss

def crossValidationForLogistic_reg_with_loss(x, y, splitRatio, degrees, seed =1):
    
    x_train, y_train, x_test, y_test = split_data(x, y, splitRatio, seed)
    
    loss_tr = []
    loss_te = []
    weights = []
    degr = []
    
    index = 0
    
    plot_data = []
    
    # define parameter (just add more for loops if there are more parameters for the model)
    lambdas = np.arange(0.000001,0.3,0.1)
    gammas = np.arange(0.01,0.9,0.2)
    
    for ind, lambda_ in enumerate(lambdas):
        
        for ind_d, d in enumerate(degrees):
            
            for ind_g, gamma in enumerate(gammas):
            
                #perform polynomial feature expension
                x_test_poly = build_poly(x_test,d)
                x_train_poly = build_poly(x_train, d)
            
                #normalize data (DANGER: the test set must be normalized with the training set's mean and std)
                mean = np.mean(x_train_poly, axis =0)
                std = np.std(x_train_poly, axis = 0)
            
                #put 1 if std = 0
                std = std + (std == 0)
            
                x_train_ready = (x_train_poly - mean) / std
                x_test_ready = (x_test_poly - mean) / std
                
               
                #add bias term
                
                bias_tr = np.ones(shape=x_train.shape)
                bias_te = np.ones(shape=x_test.shape)
            
                x_train_ready = np.c_[bias_tr, x_train_ready]
                x_test_ready = np.c_[bias_te, x_test_ready]
                
           

                #Model
        
                w_star, e_tr = reg_logistic_regression(y_train, x_train_ready, lambda_, np.ones(x_test_ready.shape[1]), 100, gamma)
        
           
                degr.append(d)
        
                #map the y= -1 => y = 0
                t = np.ones(len(y_test))
                t[np.where(y_test == -1)] = 0
                
                #compute the loss on the test set
                e_te = logistic_loss(t, x_test_ready, w_star) + lambda_ * np.sum(w_star**2)
           
        
                loss_tr.append(e_tr)
                loss_te.append(e_te)
                weights.append(w_star)
                
                plot_data.append((lambda_, d, gamma, e_te))
              
                print("lambda={l:.5f},degree={deg}, gamma={ga:.5f}, Training Loss={tr}, Testing Loss={te}".format(
                       l=lambda_, tr=loss_tr[index], te=loss_te[index], deg=d, ga=gamma))
        
                #increment index
                index = index + 1
    
    return weights[np.argmin(loss_te)], degr[np.argmin(loss_te)], loss_te[np.argmin(loss_te)], x_train, plot_data
    
    
