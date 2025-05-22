from preprocessing import scale_and_merge, rescale_training_and_test_sets
import torch
import numpy as np
from algorithms import ridge_regression, ridge_regression_low_rank, \
                        train_robust_weights, train_trace_norm, \
                        train_robust_weights_trace_norm, compute_weights

def leave_one_out_single(model_out,x,y,means,vars,\
                         lon_size,lat_size,notnan_idx,nan_idx,time_period=34,\
                         method='ridge',rank=5,lambda_=1.0,mu_=1.0,nu_=1.0,\
                         lr=1e-5,nb_gradient_iterations=50,dtype=torch.float32,verbose=False):
    
    """Run a single iteration the leave-one-out procedure (LOO) with model_out out of the training set.

        Args:

        Returns:
    """
    w = torch.zeros(lon_size * lat_size, lon_size * lat_size,dtype=dtype)
    training_models, x_rescaled, y_rescaled = rescale_training_and_test_sets(model_out,x,y,means,vars,dtype=dtype)
    # _, x_train_merged, y_train_merged, x_test_merged, y_test_merged = scale_and_merge(model_out,x_rescaled,y_rescaled,means,vars,dtype=torch.float32)

    # training_models, x_rescaled, y_rescaled = rescale_training_and_test_sets(model_out,x,y,means,vars,dtype=dtype)
    _,  x_train_merged, y_train_merged, x_test_merged, y_test_merged = scale_and_merge(model_out,x_rescaled,y_rescaled,means,vars,dtype=dtype)

    # if method = ridge, then we train the ridge regression model
    if (method == 'ridge') and (rank is None):

        # compute ridge regression coefficient 
        w[np.ix_(notnan_idx,notnan_idx)] = ridge_regression(x_train_merged[:,notnan_idx], y_train_merged[:,notnan_idx], len(training_models)*lambda_,dtype=dtype,verbose=verbose)

    elif (method == 'ridge') and (rank is not None):

        # compute low rank ridge regression coefficient
        w[np.ix_(notnan_idx,notnan_idx)] = ridge_regression_low_rank(x_train_merged[:,notnan_idx], y_train_merged[:,notnan_idx], rank,  len(training_models)*lambda_,dtype=dtype,verbose=verbose)

    elif method == 'robust':

        # compute low rank ridge regression coefficient
        w, loss  = train_robust_weights(training_models,x,y,lon_size,lat_size,notnan_idx,rank,lambda_,mu_,lr,nb_iterations=nb_gradient_iterations,dtype=dtype,verbose=verbose)

    elif method == 'trace_norm':

        # compute trace norm and ridge regression coefficient
        w[np.ix_(notnan_idx,notnan_idx)]  = train_trace_norm(x_train_merged[:,notnan_idx], y_train_merged[:,notnan_idx],  len(training_models)*lambda_, nu_, dtype=dtype,verbose=verbose)

    elif method == 'robust_trace_norm':

        # compute trace norm and ridge regression coefficient
        w  = train_robust_weights_trace_norm(training_models,x,y,lon_size,lat_size,notnan_idx,lambda_,mu_,nu_,lr,nb_iterations=nb_gradient_iterations,dtype=dtype,verbose=verbose)

    # Predictions on test set
    y_pred = torch.zeros_like(x[model_out],dtype=dtype)
    y_pred[:,:,nan_idx] = float('nan')
    y_pred[:,:,notnan_idx] = x[model_out][:,:,notnan_idx] @ w[np.ix_(notnan_idx,notnan_idx)]


    # Compute training errors
    y_pred_train = {}
    rmse_train = {m: 0 for m in training_models}

    if verbose == True:
        for idx_m,m in enumerate(x.keys()):
            
            if m != model_out:

                y_pred_train[m] = torch.zeros(x[m].shape[0],time_period,lon_size*lat_size,dtype=dtype)
                y_pred_train[m][:,:,notnan_idx] =  x[m][:,:,notnan_idx] @ w[np.ix_(notnan_idx,notnan_idx)]
                rmse_train[m] = torch.mean(torch.norm(y_pred_train[m][:,:,notnan_idx] - y[m][:,:,notnan_idx],dim=(1,2) )**2/torch.norm(y[m][:,:,notnan_idx],dim=(1,2))**2)
                                           
    return w, y_pred, y[model_out], rmse_train



def leave_one_out_procedure(x,y,means,vars,\
                            lon_size,lat_size, notnan_idx, nan_idx,time_period=34,\
                            method='ridge',rank=None,lambda_=1.0,mu_=1.0,nu_=1.0,\
                            lr=1e-5,nb_gradient_iterations=20,dtype=torch.float32,verbose=False):
    """It runs the LOO procedure.

    """
    w = {}
    y_pred = {}
    y_test = {}
    
    mse_mean = {}
    
    weights = {m: 0.0 for idx_m, m in enumerate(x.keys())}
    training_loss = {m: {} for idx_m, m in enumerate(x.keys())}
    
    for idx_m, m in enumerate(x.keys()):

        # run leave one out
        w[m], y_pred[m], y_test[m], training_loss[m] = leave_one_out_single(m,x,y,means,vars,\
                                                            lon_size,lat_size,notnan_idx,nan_idx,time_period,\
                                                            method,rank,lambda_,mu_,nu_,\
                                                            lr,nb_gradient_iterations,dtype=dtype,verbose=verbose)

        
        # compute the mean rmse 
        # mse_mean[m] = torch.mean((y_pred[m][:,:,notnan_idx] - y_test[m][:,:,notnan_idx])**2 / vars[m][:,:,notnan_idx],dtype=dtype) 
        mse_mean[m] = torch.mean( torch.norm(y_pred[m][:,:,notnan_idx] - y_test[m][:,:,notnan_idx], dim=(1,2))**2 /torch.norm(y_test[m][:,:,notnan_idx],dim=(1,2))**2 , dtype=dtype)        
    
        # print the rmse
        print('RMSE (mean) on model ', m, ' : ', mse_mean[m].item())

        # list of training models
        models_tmp = list(x.keys())
        models_tmp.remove(m)

        if (method == 'robust') or (method=='robust_trace_norm'):

            # compute robust model weights
            weights[m] = compute_weights(models_tmp,w[m],x,y,notnan_idx,mu_, dtype=dtype) 
                                    
        else:

            # if we do not use the robust weight approach, then we compute uniform weights
            weights[m] = {m_tmp: 1/len(models_tmp) for m_tmp in models_tmp}
            
        # compute weight = 0.0 for climate model m 
        weights[m][m] = 0.0

    weights_tmp = torch.zeros(len(x.keys()),dtype=dtype)
    for idx_m, m in enumerate(x.keys()):
        weights_tmp +=  torch.tensor(list(weights[m].values()),dtype=dtype)

    weights_tmp = weights_tmp / len(x.keys())

    # Check that the sum of weights = 1

    if torch.sum(weights_tmp,dtype=dtype).item() != 1.0:
        print('Warning: Sum of weights is not equal to 1.0 : ', torch.sum(weights_tmp,dtype=dtype).item())
    else:
        print("Sum of weights ==1 : ", torch.sum(weights_tmp,dtype=dtype).item())

    return w, mse_mean, weights, training_loss