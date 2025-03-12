# scripts that runs cross validation procedure
import os
import sys
maindir = os.getcwd()
sys.path.append(maindir+"/src")

import torch # type: ignore
import numpy as np
import pickle
from preprocessing import data_processing, compute_anomalies_and_scalers, \
                            compute_forced_response, numpy_to_torch

from cross_validation import cross_validation_procedure


DTYPE = torch.float32

############### Load climate model raw data for SST
with open('data/ssp585_time_series.pkl', 'rb') as f:
    data = pickle.load(f)

# define range of values
# mu_range_tmp = np.array([1.0, 5.0, 10.0, 50, 100.0,500.0, 1000.0])
# lambda_range_tmp = np.array([0.0, 10.0,50.0,100.0,500.0, 1000.0, 5000.0, 10000.0])

mu_range_tmp = np.array([1000.0])
lambda_range_tmp = np.array([100.0])
nu_range_tmp = np.array([10.0])

with open('data/mu_range_bis.npy', 'wb') as f:
    np.save(f, mu_range_tmp)

with open('data/lambda_range_bis.npy', 'wb') as f:
    np.save(f, lambda_range_tmp)

###################### Load longitude and latitude 
with open('data/lon.npy', 'rb') as f:
    lon = np.load(f)

with open('data/lat.npy', 'rb') as f:
    lat = np.load(f)


# define grid (+ croping for latitude > 60)
lat_grid, lon_grid = np.meshgrid(lat[lat<=60], lon, indexing='ij')

lat_size = lat_grid.shape[0]
lon_size = lon_grid.shape[1]


##################### Preprocess data : get x and y 
data_processed, notnan_idx, nan_idx = data_processing(data, lon, lat,max_models=100)
x, means, vars = compute_anomalies_and_scalers(data_processed, lon_size, lat_size, nan_idx, time_period=34)
y = compute_forced_response(data_processed, lon_size, lat_size, nan_idx, time_period=34)

x,y, means, vars = numpy_to_torch(x,y,means,vars, dtype=DTYPE)
###############################################################################

########################## get ridge regressor W
w_ridge, rmse_ridge, weights_ridge, training_loss_ridge = \
                        cross_validation_procedure(x,y,means,vars,\
                                                    lon_size,lat_size,notnan_idx,nan_idx,time_period=34,\
                                                    method='trace_ridge', rank=None, lambda_range=lambda_range_tmp, mu_range = mu_range_tmp, nu_range = nu_range_tmp ,\
                                                    lr=1e-5,nb_gradient_iterations=20,dtype=DTYPE,verbose=True)
                           

### save files
with open('results/ridge_scaled/w_ridge.pkl', 'wb') as f:
    pickle.dump(w_ridge, f)

with open('results/ridge_scaled/rmse_ridge.pkl', 'wb') as f:
    pickle.dump(rmse_ridge, f)

with open('results/ridge_scaled/weights_ridge.pkl', 'wb') as f:
    pickle.dump(weights_ridge, f)

with open('results/ridge_scaled/training_loss_ridge.pkl', 'wb') as f:
    pickle.dump(training_loss_ridge, f)

######################### get ridge regressor W low-rank
w_ridge_lr, rmse_ridge_lr, weights_ridge_lr, training_loss_ridge_lr = \
                            cross_validation_procedure(x,y,means,vars,\
                                                    lon_size,lat_size,notnan_idx,nan_idx,time_period=34,\
                                                    method='ridge', rank=20, lambda_range=lambda_range_tmp, mu_range = mu_range_tmp ,\
                                                    lr=1e-5,nb_gradient_iterations=20,dtype=DTYPE,verbose=True)


### save files
with open('results/ridge_low_rank_scaled/w_ridge_lr.pkl', 'wb') as f:
    pickle.dump(w_ridge_lr, f)

with open('results/ridge_low_rank_scaled/rmse_ridge_lr.pkl', 'wb') as f:
    pickle.dump(rmse_ridge_lr, f)

with open('results/ridge_low_rank_scaled/weights_ridge_lr.pkl', 'wb') as f:
    pickle.dump(weights_ridge_lr, f)

with open('results/ridge_low_rank_scaled/training_loss_ridge_lr.pkl', 'wb') as f:
    pickle.dump(training_loss_ridge_lr, f)



####################### get trace norm regressor W
w_trace, rmse_trace, weights_trace, training_loss_trace = \
                        cross_validation_procedure(x,y,means,vars,\
                        lon_size,lat_size,notnan_idx,nan_idx,time_period=34,\
                        method='trace_norm', rank=None, lambda_range=lambda_range_tmp, mu_range = mu_range_tmp, nu_range=nu_range_tmp ,\
                        lr=1e-5,nb_gradient_iterations=20,dtype=DTYPE,verbose=True)

## save files
with open('results/trace_norm_scaled/w_trace.pkl', 'wb') as f:
    pickle.dump(w_trace, f)

with open('results/trace_norm_scaled/rmse_trace.pkl', 'wb') as f:
    pickle.dump(rmse_trace, f)

with open('results/trace_norm_scaled/weights_trace.pkl', 'wb') as f:
    pickle.dump(weights_trace, f)

with open('results/trace_norm_scaled/training_loss_trace.pkl', 'wb') as f:
    pickle.dump(training_loss_trace, f)

# ############################ get robust regressor W 
# w_robust, rmse_robust, weights_robust, training_loss_robust = \
#                         cross_validation_procedure(x,y,vars,\
#                         lon_size,lat_size,notnan_idx,nan_idx,time_period=33,\
#                         method='robust', rank=None, lambda_range=lambda_range_tmp, mu_range = mu_range_tmp ,\
#                         lr=1e-6,nb_gradient_iterations=100,dtype=DTYPE,verbose=True)

# ## save files
# with open('results/robust_scaled/w_robust_bis.pkl', 'wb') as f:
#     pickle.dump(w_robust, f)

# with open('results/robust_scaled/rmse_robust_bis.pkl', 'wb') as f:
#     pickle.dump(rmse_robust, f)

# with open('results/robust_scaled/weights_robust_bis.pkl', 'wb') as f:
#     pickle.dump(weights_robust, f)

# with open('results/robust_scaled/training_loss_robust_bis.pkl', 'wb') as f:
#     pickle.dump(training_loss_robust, f)


### get robust regressor W low-rank
# w_robust_lr, rmse_robust_lr, weights_robust_lr,training_loss_robust_lr = \
#                             cross_validation_procedure(x,y,vars,\
#                             lon_size,lat_size,notnan_idx,nan_idx,time_period=33,\
#                             method='robust', rank=20, lambda_range=lambda_range_tmp, mu_range = mu_range_tmp ,\
#                             lr=1e-6,nb_gradient_iterations=100,dtype=DTYPE,verbose=False)



# with open('results/robust_low_rank_scaled/w_robust_lr_bis.pkl', 'wb') as f:
#     pickle.dump(w_robust_lr, f)

# with open('results/robust_low_rank_scaled/rmse_robust_lr_bis.pkl', 'wb') as f:
#     pickle.dump(rmse_robust_lr, f)

# with open('results/robust_low_rank_scaled/weights_robust_lr_bis.pkl', 'wb') as f:
#     pickle.dump(weights_robust_lr, f)

# with open('results/robust_low_rank_scaled/training_loss_robust_lr_bis.pkl', 'wb') as f:
#     pickle.dump(training_loss_robust_lr, f)

