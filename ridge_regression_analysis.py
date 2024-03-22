import pickle
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as netcdf
import torch


def train_ridge_regression(x,y,lon_size,lat_size,lambda_,nbEpochs=200,verbose=True):
    """
    Given a model m, learn parameter β^m such that β^m = argmin_{β}(||y_m - X_m^T β||^2) ).

    Args:
        - x, y: training set and training target 
        - lon_size, lat_size: longitude and latitude grid size (Int)
        - lambda_: regularizer coefficient (float)
        - nbepochs: number of optimization steps (Int)
        - verbose: display logs (bool)
    """

    # define variable beta
    beta = torch.zeros(lon_size*lat_size).to(torch.float64)
    beta.requires_grad_(True)  
                          
    # define optimizer
    optimizer = torch.optim.Adam([beta],lr=1e-3)
            
    # --- optimization loop ---                
    for epoch in torch.arange(nbEpochs):      
                      
        optimizer.zero_grad()
        ############### Define loss function ##############
                    
        # first term: ||Y - β^T X||
        obj = 0.5*torch.mean((y - torch.matmul(x,beta))**2)
        obj += 0.5*lambda_*torch.norm(beta,p=2)**2
                    
        #define loss function
        loss = obj
                    
        # Use autograd to compute the backward pass. 
        loss.backward(retain_graph=True)               
        
        # take a step into optimal direction of parameters minimizing loss
        optimizer.step()       
        
        if(verbose==True):
            if(epoch % 10 == 0):
                print('Epoch ', epoch.detach().item(), 
                    ', loss=', loss.detach().item()
                    )
    return beta.detach().clone()


def train_one_step_cross_validation(subset_runs,x,y,lon_size,lat_size,lambda_,nbEpochs=200,verbose=True):
    """
    Construct a training set and test set of a given 
        - x,y are the set of runs of a single model
    """
    # construct the training set
    y_train = 0
    y_test = 0
    x_train = 0
    x_test = 0
    
    for idx_r, r in enumerate(subset_runs):
        
        if idx_r ==0:
            y_train = y[r]
            x_train = x[r]
        else:
            y_train = torch.cat([y_train, y[r]])
            x_train = torch.cat([x_train, x[r]],axis=0)     
    
    beta = train_ridge_regression(x_train,y_train,lon_size,lat_size,lambda_,nbEpochs,verbose)
    
    return beta


def single_cross_validation(x,y,lon_size,lat_size,lambda_,nbEpochs=200,verbose=True):
    """
    Cross validation procedure: LOO--> for a given model, train the ridge regression on all runs except one, and test on this one.

    Args:
        - x, y: training set and training target of a single model
        - lon_size, lat_size: longitude and latitude grid size (Int)
        - lambda_: regularizer coefficient (float)
        - nbepochs: number of optimization steps (Int)
        - verbose: display logs (bool)
    """
    # number of runs
    n_runs = len(x.keys())

    beta = {}
    y_pred = {}
    rmse = {}

    for idx_r, r in enumerate(x.keys()):

        # get list of run names 
        list_runs = list(x.keys())
        
        # remove run r
        list_runs.remove(r)

        # train model 
        beta[r] = train_one_step_cross_validation(list_runs,x,y,lon_size,lat_size,lambda_,nbEpochs,verbose)

        # prediction
        y_pred[r] = torch.matmul(x[r],beta[r]) 

        # rmse 
        rmse[r] = torch.sqrt(torch.mean((y[r] - y_pred[r])**2))


    return beta, y_pred, rmse


def model_cross_validation(x,y,lon_size,lat_size,lambda_range,nbEpochs=200,verbose=True):
    """
    Cross validation procedure: LOO--> for a given model, train the ridge regression on all runs except one, and test on this one.

    Args:
        - x, y: training set and training target of a single model
        - lon_size, lat_size: longitude and latitude grid size (Int)
        - lambda_: regularizer coefficient (float)
        - nbepochs: number of optimization steps (Int)
        - verbose: display logs (bool)
    """
    beta = {}
    rmse = {}
    
    # for each model, run the single cross validation
    for idx_lambda, lambda_ in enumerate(lambda_range):

        beta[lambda_], y_pred, rmse[lambda_] = single_cross_validation(x,y,lon_size,lat_size,lambda_,nbEpochs,verbose)

    return beta, rmse
    

def all_models_cross_validation(x,y,lon_size,lat_size,lambda_range,nbEpochs=200,verbose=True):
    """
    Cross validation procedure: LOO--> for each model, train the ridge regression on all runs except one, and test on this one.

    Args:
        - x, y: training set and training target of a single model
        - lon_size, lat_size: longitude and latitude grid size (Int)
        - lambda_range: set of regularizer coefficients to test (float)
        - nbepochs: number of optimization steps (Int)
        - verbose: display logs (bool)
    """
    beta = {}
    rmse = {}

    for idx_m, m in enumerate(x.keys()):

        beta[m], rmse[m] = model_cross_validation(x[m],y[m],\
                                                lon_size,lat_size,\
                                                lambda_range,nbEpochs,verbose)

    return beta, rmse