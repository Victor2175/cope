import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
# import argparse

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "ForceSMIP"))

# Import our modules
from data_loader import ForceSMIPDataLoader, yearly_average, compute_yearly_average_dict
from preprocessing import (
    merge_training_data, reshape_training_data, capture_nans,
    apply_notnan_filter_complete,moving_average_smoothing
)
from algorithms import ridge_regression, LowRankSolver
from forcesmip_pipeline import ForceSMIPPipeline
from utils import to_xarray


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("Modules loaded successfully!")

############################# Initialize Pipeline ############################

# Initialize pipeline
base_path = '/net/krypton/climdyn_nobackup/FTP'

# variable names: pr, psl, tas, tos
variable = 'tas'  # Temperature at 2m

pipeline = ForceSMIPPipeline(base_path, variable=variable)
print(f"Pipeline initialized for variable: {variable}")


############################# Create data loader and load data ############################

# Initialize data loader
data_loader = ForceSMIPDataLoader(base_path)

# Load training data
print("Loading training data...")
dic_data, dic_forced_response, longitude, latitude = data_loader.load_training_data(variable)

print(f"Training data loaded:")
print(f"  Number of models: {len(dic_data)}")
print(f"  Model names: {list(dic_data.keys())}")
print(f"  Grid shape: {latitude.shape[0]} x {longitude.shape[0]}")
print(f"  Time steps: {list(dic_data.values())[0].shape[1]}")


# Load test data
test_models = ['1A', '1B', '1C', '1D', '1E', '1F', '1G', '1H', '1J']
print("\nLoading test data...")

data_test, temporal_means = data_loader.load_test_data(variable, tier='Tier1', test_models=test_models)


############ compute yearly averages ############

# Compute yearly averages
print("Computing yearly averages...")
dic_data_yearly = compute_yearly_average_dict(dic_data)
dic_forced_response_yearly = compute_yearly_average_dict(dic_forced_response)

data_test_yearly = yearly_average(data_test)


############## build training set ###############
# Prepare training data for machine learning
print("Preparing training data...")

# Method 1: Merge all training data
x_train_merged, y_train_merged = merge_training_data(
    dic_data_yearly, dic_forced_response_yearly
)

# Method 2: Keep model separation
x_train_dict, y_train_dict = reshape_training_data(
    dic_data_yearly, dic_forced_response_yearly
)

print(f"Merged training data shape: X={x_train_merged.shape}, Y={y_train_merged.shape}")
print(f"Separated training data: {len(x_train_dict)} models")


# capture NaNs in training data
nan_idx, notnan_idx =capture_nans(x_train_dict)

# Prepare test data for smoothing experiments
x_test = torch.from_numpy(data_test.reshape(
    data_test.shape[0], data_test.shape[1], -1
)).to(torch.float32)

print(f"Test data shape for smoothing: {x_test.shape}")

############################################# add the nans of the test sets #############################
# record nan indices as the union of nans on each map
nan_idx_test = nan_idx.copy()

for idx_r in range(x_test.shape[0]):

    # get nan mask of test set 
    nan_mask = np.where(np.abs(x_test[idx_r,:,:])>1e10, True, False)

    if nan_mask.any() == False:
        nan_mask = np.where(np.isnan(x_test[idx_r,:,:])==True, True, False)

    # get the index of columns where there is at least one True in the nan mask
    col_indices = np.where(np.any(nan_mask, axis=0))[0]

    nan_idx_test = list(set(nan_idx_test) | set(col_indices))    

# define not nan indices (useful to ease the computations)
notnan_idx_test = list(set(list(range(x_test.shape[2]))) - set(nan_idx_test))
    

# # replace the nan index by Nans
if variable != 'tas' and variable != 'pr':
    notnan_idx = notnan_idx_test
    nan_idx = nan_idx_test


################################## Transform the data and keep only the not nan indices ##################################
# Apply not-nan filter to training data
x_train_merged_filtered, y_train_merged_filtered, \
x_train_dict_filtered, y_train_dict_filtered, x_test_filtered = apply_notnan_filter_complete(
    x_train_merged, y_train_merged,
    x_train_dict, y_train_dict,
    x_test, notnan_idx
)

####### If it is tas variabe, then set the values > 1e9 to 0.0 ######
if variable == 'tas' or variable == 'pr':
    x_test_filtered[np.abs(x_test_filtered)>1e9] = 0.0

################################### Add some smoothing on the test data ####################################################
x_test_smooth = moving_average_smoothing(x_test_filtered, window_size=12*30, mode='same')
print(f"Test data shape after filtering and smoothing: {x_test_smooth.shape}")

## 6.5 Cross-Validation to get the best Lambda Optimization

print("="*70)
print("============== CROSS-VALIDATION LAMBDA OPTIMIZATION =====================")
print("="*70)

# Import the cross-validation function
from algorithms import cross_validation_lambda_optimization, cross_validation_lambda_rank_optimization,\
                        plot_cv_lambda_results, plot_cv_lambda_rank_results


# Define lambda values to test (broader range for better optimization)
lambda_values_cv = [10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0, 500000.0, 1000000.0]
rank_values_cv = [2, 5, 10, 20, 30, 50, 100]

lambda_values_cv = [1000.0]
rank_values_cv = [10]


print("Performing cross-validation to optimize lambda for worst-case performance...")

##################### Optimize lambda using cross-validation with worst-case objective
cv_optimization = cross_validation_lambda_optimization(
    x_train_dict_filtered, y_train_dict_filtered,
    lambda_values=lambda_values_cv,
    cv_folds=5,
    objective='worst_case',  # Optimize for worst-case NRMSE
    verbose=True
)


# Extract optimal lambda
optimal_lambda_cv = cv_optimization['best_lambda']
print(f"\n🎯 OPTIMAL LAMBDA (CV-optimized): {optimal_lambda_cv}")
print(f"   Worst-case NRMSE: {cv_optimization['cv_results'][optimal_lambda_cv]['worst_nrmse']:.4f}")
print(f"   Mean NRMSE: {cv_optimization['cv_results'][optimal_lambda_cv]['mean_nrmse']:.4f}")


##################### Optimize lambda and rank using cross-validation with worst-case objective
print("\nPerforming cross-validation to optimize lambda and rank for worst-case performance...")


# Optimize lambda and rank using cross-validation with worst-case objective
cv_optimization_lr = cross_validation_lambda_rank_optimization(
    x_train_dict_filtered, y_train_dict_filtered,
    lambda_values=lambda_values_cv,
    rank_values=rank_values_cv,
    cv_folds=5,
    objective='worst_case',  # Optimize for worst-case NRMSE
    verbose=True
)


# Extract optimal lambda
optimal_lambda_cv_lr = cv_optimization_lr['best_lambda']
optimal_rank_cv_lr = cv_optimization_lr['best_rank']
print(f"\n🎯 OPTIMAL LAMBDA (CV-optimized): {optimal_lambda_cv_lr}")
print(f"🎯 OPTIMAL RANK (CV-optimized): {optimal_rank_cv_lr}")
print(f"   Worst-case NRMSE: {cv_optimization_lr['cv_results'][(optimal_lambda_cv_lr, optimal_rank_cv_lr)]['worst_nrmse']:.4f}")
print(f"   Mean NRMSE: {cv_optimization_lr['cv_results'][(optimal_lambda_cv_lr, optimal_rank_cv_lr)]['mean_nrmse']:.4f}")

###############

# given the best lambda value, run the ridge regression again
w_ridge_cv = ridge_regression(x_train_merged_filtered,y_train_merged_filtered, optimal_lambda_cv, verbose=True)


# given the best lambda and best rank, run the low-rank solver again
w_ridge_lr_cv_tmp =  ridge_regression(x_train_merged_filtered,y_train_merged_filtered, optimal_lambda_cv_lr, verbose=True)


# get low rank solution
I_p = torch.eye(x_train_merged_filtered.shape[1], dtype=x_train_merged_filtered.dtype, device=x_train_merged_filtered.device)
x_augmented = torch.cat([x_train_merged_filtered, torch.sqrt(torch.tensor(optimal_lambda_cv_lr, dtype=x_train_merged_filtered.dtype, device=x_train_merged_filtered.device)) * I_p], dim=0)
U, S, Vt = torch.linalg.svd(x_augmented @ w_ridge_lr_cv_tmp, full_matrices=False)

w_ridge_lr_cv = w_ridge_lr_cv_tmp @ Vt[:optimal_rank_cv_lr, :].T @ Vt[:optimal_rank_cv_lr, :]

# Make predictions using different methods

# reshape temporal means
temporal_means = temporal_means.reshape(temporal_means.shape[0], temporal_means.shape[1], -1)
temporal_means = temporal_means[:,:,notnan_idx]

print("Making predictions...")
# Global models
y_pred_ridge = x_test_filtered @ w_ridge_cv + temporal_means
y_pred_ridge_fullmap = torch.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
y_pred_ridge_fullmap[:, :, notnan_idx] = y_pred_ridge
y_pred_ridge_fullmap[:, :, nan_idx] = float('nan')
y_pred_ridge_fullmap = y_pred_ridge_fullmap.reshape(x_test.shape[0], x_test.shape[1], latitude.shape[0], longitude.shape[0])


y_pred_ridge_lr = x_test_filtered @ w_ridge_lr_cv + temporal_means
y_pred_ridge_lr_fullmap = torch.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
y_pred_ridge_lr_fullmap[:, :, notnan_idx] = y_pred_ridge_lr
y_pred_ridge_lr_fullmap[:, :, nan_idx] = float('nan')
y_pred_ridge_lr_fullmap = y_pred_ridge_lr_fullmap.reshape(x_test.shape[0], x_test.shape[1], latitude.shape[0], longitude.shape[0])



y_pred_ridge_smooth = x_test_smooth @ w_ridge_cv + temporal_means
y_pred_ridge_smooth_fullmap = torch.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
y_pred_ridge_smooth_fullmap[:, :, notnan_idx] = y_pred_ridge_smooth
y_pred_ridge_smooth_fullmap[:, :, nan_idx] = float('nan')
y_pred_ridge_smooth_fullmap = y_pred_ridge_smooth_fullmap.reshape(x_test.shape[0], x_test.shape[1], latitude.shape[0], longitude.shape[0])



y_pred_ridge_lr_smooth = x_test_smooth @ w_ridge_lr_cv + temporal_means
y_pred_ridge_lr_smooth_fullmap = torch.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
y_pred_ridge_lr_smooth_fullmap[:, :, notnan_idx] = y_pred_ridge_lr_smooth
y_pred_ridge_lr_smooth_fullmap[:, :, nan_idx] = float('nan')
y_pred_ridge_lr_smooth_fullmap = y_pred_ridge_lr_smooth_fullmap.reshape(x_test.shape[0], x_test.shape[1], latitude.shape[0], longitude.shape[0])



# save the prediction into numpy array
y_pred_ridge_xr, lon_xr, lat_xr = to_xarray(y_pred_ridge_fullmap, test_models, longitude, latitude)
y_pred_ridge_xr.to_dataset(name='prediction')
y_pred_ridge_lr_xr, lon_xr, lat_xr = to_xarray(y_pred_ridge_lr_fullmap, test_models, longitude, latitude)
y_pred_ridge_lr_xr.to_dataset(name='prediction')

y_pred_ridge_smooth_xr, lon_xr, lat_xr = to_xarray(y_pred_ridge_smooth_fullmap, test_models, longitude, latitude)
y_pred_ridge_smooth_xr.to_dataset(name='prediction')
y_pred_ridge_lr_smooth_xr, lon_xr, lat_xr = to_xarray(y_pred_ridge_lr_smooth_fullmap, test_models, longitude, latitude)
y_pred_ridge_lr_smooth_xr.to_dataset(name='prediction')


# save it as a netcdf file
path_tier1 = '/home/vcohen/cope/results/tier1/'

y_pred_ridge_xr.to_netcdf(f'{path_tier1}predictions_{variable}_ridge_tier1.nc')
y_pred_ridge_lr_xr.to_netcdf(f'{path_tier1}predictions_{variable}_ridge_lr_tier1.nc')
y_pred_ridge_smooth_xr.to_netcdf(f'{path_tier1}predictions_{variable}_ridge_smooth_tier1.nc')
y_pred_ridge_lr_smooth_xr.to_netcdf(f'{path_tier1}predictions_{variable}_ridge_lr_smooth_tier1.nc')