import torch
import numpy as np
from leave_one_out import leave_one_out_procedure

def cross_validation_procedure(x,y,vars,\
                               lon_size,lat_size,notnan_idx,nan_idx,time_period=33,\
                               method='ridge', rank=None, lambda_range=torch.tensor([1.0]), mu_range=torch.tensor([1.0]),\
                               lr=1e-5,nb_gradient_iterations=20,verbose=True):
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

        for idx_mu, mu_ in enumerate(mu_range):

            w_tmp, rmse[(lambda_,mu_)], weights[(lambda_,mu_)] = leave_one_out_procedure(x,y,vars,\
                                                                                        lon_size,lat_size, notnan_idx,nan_idx,time_period,\
                                                                                        method, rank, lambda_, mu_,\
                                                                                        lr,nb_gradient_iterations,verbose)
            
            # w_tmp is very big: we take the mean of it
            w[(lambda_,mu_)] = torch.from_numpy(np.nanmean(np.array(list(w_tmp.values())), axis=0))
            

    return w,rmse,weights
    