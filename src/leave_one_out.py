from preprocessing import build_training_and_test_sets
import torch
import numpy as np
from algorithms import ridge_regression, ridge_regression_low_rank, train_robust_weights_model, compute_weights

def leave_one_out_single(model_out,x,y,vars,\
                         lon_size,lat_size,notnan_idx,nan_idx,\
                         lr=0.0001,nb_gradient_iterations=50,\
                         time_period=33,rank=5,\
                         lambda_=1.0,method='ridge',mu_=1.0,verbose=True):
    """Run a single iteration the leave-one-out procedure (LOO) with model_out out of the training set.

        Args:

        Returns:
    """
    w = torch.zeros(lon_size * lat_size, lon_size * lat_size).to(torch.float64)
    training_models, x_train, y_train, x_test, y_test = build_training_and_test_sets(model_out,x,y,vars,lon_size,lat_size,time_period=33)

    # if method = ridge, then we train the ridge regression model
    if method == 'ridge':

        # compute ridge regression coefficient 
        w[np.ix_(notnan_idx,notnan_idx)] = ridge_regression(x_train[:,notnan_idx], y_train[:,notnan_idx], lambda_)

    elif method == 'ridge-lr':

        # compute low rank ridge regression coefficient
        w[np.ix_(notnan_idx,notnan_idx)] = ridge_regression_low_rank(x_train[:,notnan_idx], y_train[:,notnan_idx], rank, lambda_)

    elif method == 'robust':

        # compute low rank ridge regression coefficient
        w, training_loss  = train_robust_weights_model(training_models,x,y,lon_size,lat_size,notnan_idx,rank,lambda_,mu_,lr =0.00001,nb_iterations=nb_gradient_iterations)

    # Predictions on test set
    y_pred = torch.ones_like(x_test).to(torch.float64)
    y_pred[:,nan_idx] = float('nan')
    y_pred[:,notnan_idx] = x_test[:,notnan_idx].to(torch.float64) @ w[np.ix_(notnan_idx,notnan_idx)].to(torch.float64)

    # Compute training errors
    y_pred_train = {}
    rmse_train = {}

    for idx_m,m in enumerate(x.keys()):
        if m != model_out:
            y_pred_train[m] = {}
            rmse_train[m] = 0.0
            for idx_r, r in enumerate(x[m].keys()):
                y_pred_train[m][r] = torch.zeros(time_period,lon_size*lat_size).to(torch.float64)
                y_pred_train[m][r][:,notnan_idx] =  x[m][r][:,notnan_idx] @ w[np.ix_(notnan_idx,notnan_idx)].to(torch.float64)
                rmse_train[m] += torch.nanmean((y_pred_train[m][r] - y[m][r])**2)
            rmse_train[m] = rmse_train[m] /len(x[m].keys())


    # compute the weights
    if method == "robust":
        weights = compute_weights(training_models,w,x,y,notnan_idx,lambda_,mu_)
    else:
        weights = (1/len(training_models)) * torch.ones(len(training_models))
    
    return w, y_pred, y_test, rmse_train



def leave_one_out_procedure(x,y,vars,\
                            lon_size,lat_size, notnan_idx,nan_idx,\
                            lr=0.00001,nb_gradient_iterations=20,time_period=33,\
                            rank=5,lambda_=1.0,method='ridge',mu_=1.0,verbose=True):
    """It runs the LOO procedure.

    """
    w = {}
    y_pred = {}
    y_test = {}
    
    rmse_mean = {}
    
    weights = {m: 0.0 for idx_m, m in enumerate(x.keys())}
    training_loss = {m: {} for idx_m, m in enumerate(x.keys())}
    
    for idx_m, m in enumerate(x.keys()):

        # run leave one out
        w[m], y_pred[m], y_test[m], training_loss[m] = leave_one_out_single(m,x,y,vars,\
                                                                            lon_size,lat_size,notnan_idx,nan_idx,\
                                                                            lr,nb_gradient_iterations,\
                                                                            time_period,rank,lambda_,method,mu_,verbose)

        
        # compute mean rmse 
        rmse_mean[m] = torch.nanmean((y_pred[m] - y_test[m])**2)        
    
        # print the rmse
        print('RMSE (mean) on model ', m, ' : ', rmse_mean[m].item())

        # list of training models
        models_tmp = list(training_loss[m].keys())

        if method == 'robust':

            # compute robust model weights
            weights[m] = compute_weights(models_tmp,w[m],x,y,notnan_idx,lambda_,mu_)

        else:

            # if we do not use the robust weight approach, then we compute uniform weights
            weights[m] = {m_tmp: 1/len(models_tmp) for m_tmp in models_tmp}
            
        # compute weight = 0.0 for climate model m 
        weights[m][m] = 0.0

    weights_tmp = torch.zeros(len(x.keys()))
    for idx_m, m in enumerate(x.keys()):
        weights_tmp +=  torch.tensor(list(weights[m].values()))

    weights_tmp = weights_tmp / (len(x.keys()))

    # Check that the sum of weights = 1

    if torch.sum(weights_tmp).item() != 1.0:
        print('Warning: Sum of weights is not equal to 1.0')
    else:
        print("Sum of weights ==1 : ", torch.sum(weights_tmp).item())

    return w, rmse_mean, training_loss, weights