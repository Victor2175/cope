# scripts that runs cross validation procedure
import os
import sys
maindir = os.getcwd()
sys.path.append(maindir+"/src")

import torch
import numpy as np
import pickle
from preprocessing import data_processing, compute_anomalies, extract_longitude_latitude, \
                            compute_forced_response, compute_variance, \
                            merge_runs, numpy_to_torch, standardize, build_training_and_test_sets

from cross_validation import cross_validation_procedure


############### Load climate model raw data for SST
with open('data/ssp585_time_series.pkl', 'rb') as f:
    data = pickle.load(f)

# define range of values
mu_range_tmp = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
lambda_range_tmp = np.array([0.01,0.1,0.5, 1.0,10.0,100.0, 200.0, 300.0])

mu_range_tmp = np.array([1.0])
lambda_range_tmp = np.array([100.0])

with open('data/mu_range.npy', 'wb') as f:
    np.save(f, mu_range_tmp)

with open('data/lambda_range.npy', 'wb') as f:
    np.save(f, lambda_range_tmp)

###################### Load ERA5 data
lon, lat, lon_grid, lat_grid = extract_longitude_latitude() 
lat_size = lat_grid.shape[0]
lon_size = lon_grid.shape[1]


##################### Preprocess data : get x and y 
data_processed, notnan_idx, nan_idx = data_processing(data, lon, lat)
x = compute_anomalies(data_processed, lon_size, lat_size, nan_idx, time_period=33)
y = compute_forced_response(data_processed, lon_size, lat_size, nan_idx, time_period=33)
vars = compute_variance(x, lon_size, lat_size, nan_idx, time_period=33)

# convert numpy arrays to pytorch 
x, y, vars = numpy_to_torch(x,y,vars)

# standardize data 
x, y = standardize(x,y,vars)

# merge runs for each model
x_merged, y_merged, vars_merged = merge_runs(x,y,vars)

###############################################################################

########################### get ridge regressor W
w_ridge, rmse_ridge, training_loss_ridge, weights_ridge = \
cross_validation_procedure(x,y,vars,lon_size,lat_size,notnan_idx,nan_idx,\
                            lr=0.00001,nb_gradient_iterations=2,time_period=33,\
                            rank=5,lambda_range=lambda_range_tmp,method='ridge',mu_range=mu_range_tmp,verbose=True)

### save files
with open('results/ridge/w_ridge.pkl', 'wb') as f:
    pickle.dump(w_ridge, f)

with open('results/ridge/rmse_ridge.pkl', 'wb') as f:
    pickle.dump(rmse_ridge, f)

with open('results/ridge/weights_ridge.pkl', 'wb') as f:
    pickle.dump(weights_ridge, f)

with open('results/ridge/training_loss_ridge.pkl', 'wb') as f:
    pickle.dump(training_loss_ridge, f)

########################## get ridge regressor W low-rank
w_ridge_lr, rmse_ridge_lr, training_loss_ridge_lr, weights_ridge_lr = \
cross_validation_procedure(x,y,vars,lon_size,lat_size,notnan_idx,nan_idx,\
                            lr=0.00001,nb_gradient_iterations=2,time_period=33,\
                            rank=5,lambda_range=lambda_range_tmp,method='ridge-lr',mu_range=mu_range_tmp,verbose=True)


### save files
with open('results/ridge_low_rank/w_ridge_lr.pkl', 'wb') as f:
    pickle.dump(w_ridge_lr, f)

with open('results/ridge_low_rank/rmse_ridge_lr.pkl', 'wb') as f:
    pickle.dump(rmse_ridge_lr, f)

with open('results/ridge_low_rank/weights_ridge_lr.pkl', 'wb') as f:
    pickle.dump(weights_ridge_lr, f)

with open('results/ridge_low_rank/training_loss_ridge_lr.pkl', 'wb') as f:
    pickle.dump(training_loss_ridge_lr, f)



############################# get robust regressor W 
w_robust, rmse_robust, training_loss_robust, weights_robust = \
cross_validation_procedure(x,y,vars,lon_size,lat_size,notnan_idx,nan_idx,\
                            lr=0.00001,nb_gradient_iterations=2,time_period=33,\
                            rank=None,lambda_range=lambda_range_tmp,method='robust',mu_range=mu_range_tmp,verbose=True)

### save files
with open('results/robust/w_robust.pkl', 'wb') as f:
    pickle.dump(w_robust, f)

with open('results/robust/rmse_robust.pkl', 'wb') as f:
    pickle.dump(rmse_robust, f)

with open('results/robust/weights_robust.pkl', 'wb') as f:
    pickle.dump(weights_robust, f)

with open('results/robust/training_loss_robust.pkl', 'wb') as f:
    pickle.dump(training_loss_robust, f)


### get robust regressor W low-rank
w_robust_lr, rmse_robust_lr, training_loss_robust_lr, weights_robust_lr = \
cross_validation_procedure(x,y,vars,lon_size,lat_size,notnan_idx,nan_idx,\
                            lr=0.00001,nb_gradient_iterations=2,time_period=33,\
                            rank=10,lambda_range=lambda_range_tmp,method='robust-lr',mu_range=mu_range_tmp,verbose=True)



with open('results/robust_low_rank/w_robust_lr.pkl', 'wb') as f:
    pickle.dump(w_robust_lr, f)

with open('results/robust_low_rank/rmse_robust_lr.pkl', 'wb') as f:
    pickle.dump(rmse_robust_lr, f)

with open('results/robust_low_rank/weights_robust_lr.pkl', 'wb') as f:
    pickle.dump(weights_robust_lr, f)

with open('results/robust_low_rank/training_loss_robust_lr.pkl', 'wb') as f:
    pickle.dump(training_loss_robust_lr, f)


# ################################ display results ##########################################


# ################### Ridge regresssion ########################
# # compute the ridge loo
# rmse_ridge_tmp =  np.array(list(rmse_ridge.values()))

# # worst loo Ridge
# worst_loo_ridge = np.max(rmse_ridge_tmp)
# mean_loo_ridge = np.mean(rmse_ridge_tmp)


# # quantile 95, 90, 75
# q_loo_95_ridge = np.quantile(rmse_ridge_tmp, 0.95)
# q_loo_90_ridge = np.quantile(rmse_ridge_tmp, 0.90)
# q_loo_75_ridge = np.quantile(rmse_ridge_tmp, 0.75)
# q_loo_50_ridge = np.quantile(rmse_ridge_tmp, 0.5)


# ######################## compute the ridge rrr ######################
# rmse_ridge_lr_tmp =  np.array(list(rmse_ridge_lr.values()))

# # worst loo Ridge
# worst_loo_rrr = np.max(rmse_ridge_lr_tmp)
# mean_loo_rrr = np.mean(rmse_ridge_lr_tmp)


# # quantile 95, 90, 75
# q_loo_95_rrr = np.quantile(rmse_ridge_lr_tmp, 0.95)
# q_loo_90_rrr = np.quantile(rmse_ridge_lr_tmp, 0.90)
# q_loo_75_rrr = np.quantile(rmse_ridge_lr_tmp, 0.75)
# q_loo_50_rrr = np.quantile(rmse_ridge_lr_tmp, 0.5)


# ######################## compute the robust regression ######################
# rmse_robust_tmp =  np.array(list(rmse_robust.values()))

# # worst loo Ridge
# worst_loo_robust = np.max(rmse_robust_tmp)
# mean_loo_robust = np.mean(rmse_robust_tmp)


# # quantile 95, 90, 75
# q_loo_95_robust = np.quantile(rmse_robust_tmp, 0.95)
# q_loo_90_robust = np.quantile(rmse_robust_tmp, 0.90)
# q_loo_75_robust = np.quantile(rmse_robust_tmp, 0.75)
# q_loo_50_robust = np.quantile(rmse_robust_tmp, 0.5)

# ######################## compute the robust regression with low rank constraint ######################
# rmse_robust_lr_tmp =  np.array(list(rmse_robust_lr.values()))

# # worst loo Ridge
# worst_loo_robust_rrr = np.max(rmse_robust_lr_tmp)
# mean_loo_robust_rrr = np.mean(rmse_robust_lr_tmp)


# # quantile 95, 90, 75
# q_loo_95_robust_rrr = np.quantile(rmse_robust_lr_tmp, 0.95)
# q_loo_90_robust_rrr = np.quantile(rmse_robust_lr_tmp, 0.90)
# q_loo_75_robust_rrr = np.quantile(rmse_robust_lr_tmp, 0.75)
# q_loo_50_robust_rrr = np.quantile(rmse_robust_lr_tmp, 0.5)


# print("======= Statistics ========")
# print("\n")
# print("          Ridge   RR-Lr   Robust  Robust-Lr")
# print("Worst:    {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(worst_loo_ridge,worst_loo_rrr,worst_loo_robust, worst_loo_robust_rrr))
# print("0.95:     {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(q_loo_95_ridge,q_loo_95_rrr,q_loo_95_robust, q_loo_95_robust_rrr))
# print("0.90:     {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(q_loo_90_ridge,q_loo_90_rrr,q_loo_90_robust, q_loo_90_robust_rrr))
# print("0.75:     {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(q_loo_75_ridge,q_loo_75_rrr,q_loo_75_robust, q_loo_75_robust_rrr))
# print("Median:   {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(q_loo_50_ridge,q_loo_50_rrr,q_loo_50_robust, q_loo_50_robust_rrr))
# print("Mean:     {:.3f}   {:.3f}   {:.3f}   {:.3f}".format(mean_loo_ridge,mean_loo_rrr,mean_loo_robust, mean_loo_robust_rrr))
