import torch
import numpy as np
from leave_one_out import leave_one_out_procedure

def cross_validation_procedure(x,y,means,vars,\
                               lon_size,lat_size,notnan_idx,nan_idx,time_period=34,\
                               method='ridge', rank=None, lambda_range=torch.tensor([1.0]), mu_range=torch.tensor([1.0]), nu_range=torch.tensor([1.0]),\
                               lr=1e-5,nb_gradient_iterations=20,dtype=torch.float32,verbose=True):
    """
    Cross validation procedure: LOO--> for a given model, train the ridge regression on all runs except one, and test on this one.

    Args:

    Returns:
    
    """
    w = {}
    rmse = {}
    training_loss = {}
    weights = {}
    
    # for each model, run the single cross validation
    for idx_lambda, lambda_ in enumerate(lambda_range):

        if (method == 'robust') or (method == 'robust_trace_norm'):
            for idx_mu, mu_ in enumerate(mu_range):
                if  method == 'robust_trace_norm':
                    for idx_nu, nu_ in enumerate(nu_range):
                        w_tmp, rmse[(lambda_,mu_,nu_)], weights[(lambda_,mu_,nu_)], training_loss[(lambda_,mu_,nu_)] = \
                                            leave_one_out_procedure(x,y,means,vars,\
                                                                    lon_size,lat_size, notnan_idx,nan_idx,time_period,\
                                                                    method, rank, lambda_, mu_, nu_,\
                                                                    lr,nb_gradient_iterations,dtype=dtype,verbose=verbose)
                
                        # w_tmp is very big: we take the mean of it
                        w[(lambda_,mu_,nu_)] = torch.from_numpy(np.nanmean(np.array(list(w_tmp.values())), axis=0))
                else:

                    w_tmp, rmse[(lambda_,mu_,1.0)], weights[(lambda_,mu_,1.0)], training_loss[(lambda_,mu_,1.0)] = \
                                                leave_one_out_procedure(x,y,means,vars,\
                                                                        lon_size,lat_size, notnan_idx,nan_idx,time_period,\
                                                                        method, rank, lambda_, mu_, nu_,\
                                                                        lr,nb_gradient_iterations,dtype=dtype,verbose=verbose)
                
                    # w_tmp is very big: we take the mean of it
                    w[(lambda_,mu_,1.0)] = torch.from_numpy(np.nanmean(np.array(list(w_tmp.values())), axis=0))
        else:
            for idx_nu, nu_ in enumerate(nu_range):
                w_tmp, rmse[(lambda_,1.0,nu_)], weights[(lambda_,1.0,nu_)], training_loss[(lambda_,1.0,nu_)] = \
                                            leave_one_out_procedure(x,y,means,vars,\
                                                                    lon_size,lat_size, notnan_idx,nan_idx,time_period,\
                                                                    'ridge', rank, lambda_, mu_range[0],nu_,\
                                                                    lr,nb_gradient_iterations,dtype=dtype,verbose=verbose)
                w[(lambda_,1.0,nu_)] = torch.from_numpy(np.nanmean(np.array(list(w_tmp.values())), axis=0))
                
    return w,rmse,weights, training_loss
    