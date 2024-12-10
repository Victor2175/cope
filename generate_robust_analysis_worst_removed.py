from robust_analysis import train_ridge_regression, train_robust_model, compute_weights,\
                            leave_one_out, leave_one_out_procedure, cross_validation_loo

import pickle
import os 
import netCDF4 as netcdf
import skimage
import numpy as np
import torch 

with open('ssp585_time_series.pkl', 'rb') as f:
    dic_ssp585 = pickle.load(f)

# Get the list of all files and directories
path = "/net/atmos/data/cmip6-ng/tos/ann/g025"
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

list_model = []
list_forcing = []

for idx, file in enumerate(dir_list):

    file_split = file.split("_")
    
    # extract model names
    model_name = file_split[2]
    forcing = file_split[3]
    run_name = file_split[4]
    
    list_model.append(model_name)
    list_forcing.append(forcing)
    
model_names = list(set(list_model))
forcing_names = list(set(list_forcing))


# define the file
file = '/net/h2o/climphys3/simondi/cope-analysis/data/erss/sst_annual_g050_mean_19812014_centered.nc'

# read the dataset
file2read = netcdf.Dataset(file,'r')

# load longitude, latitude and sst monthly means
lon = np.array(file2read.variables['lon'][:])
lat = np.array(file2read.variables['lat'][:])
sst = np.array(file2read.variables['sst'])

# define grid
lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

time_period = 33
grid_lat_size = lat.shape[0]
grid_lon_size = lon.shape[0]

# first filter out the models that do not contain ensemble members 
dic_reduced_ssp585 = {}

for m in list(dic_ssp585.keys()):
    if len(dic_ssp585[m].keys()) > 2:
        dic_reduced_ssp585[m] = dic_ssp585[m].copy()
        for idx_i, i in enumerate(dic_ssp585[m].keys()):
            dic_reduced_ssp585[m][i] = skimage.transform.downscale_local_mean(dic_reduced_ssp585[m][i],(1,2,2))
            lat_size = dic_reduced_ssp585[m][i][0,:,:].shape[0]
            lon_size = dic_reduced_ssp585[m][i][0,:,:].shape[1]

######## Store Nan indices 

nan_idx = []
for idx_m,m in enumerate(dic_reduced_ssp585.keys()):
    for idx_i,i in enumerate(dic_reduced_ssp585[m].keys()):    

        nan_idx_tmp = list(np.where(np.isnan(dic_reduced_ssp585[m][i][0,:,:].ravel())==True)[0])        
        nan_idx = list(set(nan_idx) | set(nan_idx_tmp))

notnan_idx = list(set(list(range(lon_size*lat_size))) - set(nan_idx))

############################


# second, for each model we compute the anomalies 
dic_processed_ssp585 = {}

import numpy as np

for idx_m,m in enumerate(dic_reduced_ssp585.keys()):
    dic_processed_ssp585[m] = dic_reduced_ssp585[m].copy()
    
    mean_ref_ensemble = 0
    y_tmp = np.zeros((len(dic_reduced_ssp585[m].keys()),time_period, lat_size*lon_size))
    
    for idx_i, i in enumerate(dic_reduced_ssp585[m].keys()):
        y_tmp[idx_i,:,:] = dic_reduced_ssp585[m][i][131:164,:,:].copy().reshape(time_period, lat_size*lon_size)
        y_tmp[idx_i,:,nan_idx] = float('nan')
           
        if idx_i == 0:
            mean_ref_ensemble = np.nanmean(y_tmp[idx_i,:,:],axis=0)/ len(dic_reduced_ssp585[m].keys())
        else:
            mean_ref_ensemble += np.nanmean(y_tmp[idx_i,:,:],axis=0)/ len(dic_reduced_ssp585[m].keys())

    for idx_i, i in enumerate(dic_processed_ssp585[m].keys()):
        dic_processed_ssp585[m][i] = y_tmp[idx_i,:,:] - mean_ref_ensemble


# compute the forced response
dic_forced_response_ssp585 = dict({})

for idx_m,m in enumerate(dic_reduced_ssp585.keys()):
    dic_forced_response_ssp585[m] = dic_reduced_ssp585[m].copy()

    for idx_i, i in enumerate(dic_forced_response_ssp585[m].keys()):
        
        y_tmp = dic_reduced_ssp585[m][i][131:164,:,:].copy().reshape(time_period, lat_size*lon_size)
        y_tmp[:,nan_idx] = float('nan')
        
        if idx_i == 0:
            mean_spatial_ensemble = np.nanmean(y_tmp,axis=1)/ len(dic_forced_response_ssp585[m].keys())
        else:
            mean_spatial_ensemble += np.nanmean(y_tmp,axis=1)/ len(dic_forced_response_ssp585[m].keys())

    for idx_i, i in enumerate(dic_forced_response_ssp585[m].keys()):  
        dic_forced_response_ssp585[m][i] = mean_spatial_ensemble - np.nanmean(mean_spatial_ensemble)

y_forced_response = {}
x_predictor = {}

for idx_m,m in enumerate(dic_processed_ssp585.keys()):
    y_forced_response[m] = {}
    x_predictor[m] = {}

    for idx_i, i in enumerate(dic_forced_response_ssp585[m].keys()):       
        y_forced_response[m][i] = dic_forced_response_ssp585[m][i]
        x_predictor[m][i] = dic_processed_ssp585[m][i]
        x_predictor[m][i][:,nan_idx] = float('nan')


# compute the variance
variance_processed_ssp585 = {}
std_processed_ssp585 = {}
for idx_m,m in enumerate(x_predictor.keys()):
    variance_processed_ssp585[m] = {}
    arr_tmp = np.zeros((len(x_predictor[m].keys()),33))
    
    for idx_i, i in enumerate(list(x_predictor[m].keys())):
        arr_tmp[idx_i,:] = np.nanmean(x_predictor[m][i],axis=1)

    arr_tmp_values = np.zeros((len(x_predictor[m].keys()),33))
    for idx_i, i in enumerate(x_predictor[m].keys()):
        arr_tmp_values[idx_i,:] = (y_forced_response[m][i] - arr_tmp[idx_i,:])**2

    variance_processed_ssp585[m] = torch.nanmean(torch.from_numpy(arr_tmp_values),axis=0)
    # variance_processed_ssp585[m] = torch.mean(torch.nanmean(torch.from_numpy(arr_tmp_values),axis=0))


# Data preprocessing
x_train = {}
y_train = {}

for idx_m,m in enumerate(dic_reduced_ssp585.keys()):
    x_train[m] = {}
    y_train[m] = {}
    for idx_i, i in enumerate(dic_processed_ssp585[m].keys()):
        x_train[m][i] = torch.nan_to_num(torch.from_numpy(x_predictor[m][i])).to(torch.float64)
        y_train[m][i] = torch.from_numpy(y_forced_response[m][i]).to(torch.float64)

mu_range = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
lambda_range = np.array([0.01,0.1,0.5, 1.0,10.0,100.0, 200.0, 300.0])

with open('mu_range_worst_removed.npy', 'wb') as f:
    np.save(f, mu_range)

with open('lambda_range_worst_removed.npy', 'wb') as f:
    np.save(f, lambda_range)



############################### filter out the worst model ################################
worst_model = 'FIO-ESM-2-0'

x_train.pop(worst_model)
x_predictor.pop(worst_model)
y_train.pop(worst_model)
y_forced_response.pop(worst_model)

################## Run the robust regression #############################

beta_robust, rmse_robust_mean, rmse_robust_q95, rmse_robust_worst, weights_robust, training_loss_robust = cross_validation_loo(x_predictor,y_forced_response,variance_processed_ssp585,\
                                                                    grid_lon_size,grid_lat_size,\
                                                                    lambda_range,'robust',mu_range,\
                                                                    nbEpochs=200,verbose=False)

rmse_robust = {'mean': rmse_robust_mean, 'q95': rmse_robust_q95, 'worst': rmse_robust_worst}

with open('results/beta_robust_worst_removed_adapted.pkl', 'wb') as f:
    pickle.dump(beta_robust, f)

with open('results/rmse_robust_worst_removed_adapted.pkl', 'wb') as f:
    pickle.dump(rmse_robust, f)

with open('results/weight_robust_worst_removed_adapted.pkl', 'wb') as f:
    pickle.dump(weights_robust, f)

with open('results/training_loss_robust_worst_removed_adapted.pkl', 'wb') as f:
    pickle.dump(training_loss_robust, f)


################### Run the ridge regressions #################################

beta_ridge, rmse_ridge_mean, rmse_ridge_q95, rmse_ridge_worst, weights_ridge, training_loss_ridge = cross_validation_loo(x_predictor,y_forced_response,variance_processed_ssp585,\
                                                            grid_lon_size,grid_lat_size,\
                                                            lambda_range,'ridge',mu_range,\
                                                            nbEpochs=200,verbose=False)

rmse_ridge = {'mean': rmse_ridge_mean, 'q95': rmse_ridge_q95, 'worst': rmse_ridge_worst}

with open('results/beta_ridge_worst_removed_adapted.pkl', 'wb') as f:
    pickle.dump(beta_ridge, f)

with open('results/rmse_ridge_worst_removed_adapted.pkl', 'wb') as f:
    pickle.dump(rmse_ridge, f)

with open('results/weights_ridge_worst_removed_adapted.pkl', 'wb') as f:
    pickle.dump(weights_ridge, f)

with open('results/training_loss_ridge_worst_removed_adapted.pkl', 'wb') as f:
    pickle.dump(training_loss_ridge, f)
        
