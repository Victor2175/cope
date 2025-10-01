# Tool functions to preprocessed the data from pkl dictionary to centered and standardized data

# Upscaling/Downscaling library
import skimage
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List


def yearly_average(x: np.ndarray) -> np.ndarray:
    """
    Compute yearly averages from monthly data without dropping spatial dims.

    Accepted shapes:
      (n_samples, n_months, lat, lon)
      (n_samples, n_months, spatial)  -> returns (n_samples, n_years, spatial)
    """
    if x.ndim == 4:
        n_samples, n_months, lat, lon = x.shape
        n_years = n_months // 12
        if n_years == 0:
            raise ValueError(f"Not enough months ({n_months}) for yearly averaging.")
        trimmed = x[:, : n_years * 12]
        return trimmed.reshape(n_samples, n_years, 12, lat, lon).mean(axis=2)
    elif x.ndim == 3:
        n_samples, n_months, spatial = x.shape
        n_years = n_months // 12
        if n_years == 0:
            raise ValueError(f"Not enough months ({n_months}) for yearly averaging.")
        trimmed = x[:, : n_years * 12]
        return trimmed.reshape(n_samples, n_years, 12, spatial).mean(axis=2)
    else:
        raise ValueError(f"Unsupported shape for yearly_average: {x.shape}")

# Deprecated shim (was removed internally)
def compute_yearly_average_dict(dic):
    """
    Deprecated: replace with {k: yearly_average(v) for k, v in dic.items()}.
    Provided only for backward compatibility with older code / tests.
    """
    return {k: yearly_average(v) for k, v in dic.items()}


def data_processing(data,longitude,latitude,max_models = 15):
    """ Process the data: statically relevant climate models (nb_runs > 3),
                            upscaling (from 0.25 to 0.5), 
                            cropping (remove latitude > 60),
                            time period focus (from 1981 until 2019)

        Args: 
            data: Dictionary of raw data (indexed by models and subdictionary indexed by runs)
            longitude, latitude: np.array, longitude and latitude coordinates
            max_models: integer, maximum number of climate models (default is 15)
            
        
        Returns:
            data_processed: upscaled data
            notnan_idx, nan_idx: non nan indices and nan indices i
    """
    # first filter out the models that contain less than 3 ensemble members 
    data_processed = {}

    # record nan indices as the union of nans on each map
    nan_idx = []


    for idx_m,m in enumerate(data.keys()):
        
        if (len(data[m].keys()) > 4) and (idx_m < max_models):

            data_processed[m] = data[m].copy()
            
            for idx_r, r in enumerate(data[m].keys()):

                # Upscaling of raw data 
                data_processed[m][r] = skimage.transform.downscale_local_mean(data_processed[m][r][:,:,:],(1,2,2))
                data_processed[m][r] = data_processed[m][r][131:,:,:]


                # capture nan indices and record the union of nans
                nan_idx_tmp = list(np.where(np.isnan(data_processed[m][r][0,:,:].ravel())==True)[0])
                nan_idx = list(set(nan_idx) | set(nan_idx_tmp))

    # get longitude and latitude size
    lon_size = longitude.shape[0]
    lat_size = latitude.shape[0]    

    # define not nan indices (useful to ease the computations)
    notnan_idx = list(set(list(range(lon_size*lat_size))) - set(nan_idx))

    return data_processed, notnan_idx, nan_idx

#######  compute anomaly scaler and forced response ########
def compute_anomalies_and_scalers(data, lon_size, lat_size, nan_idx, time_period=34):
    """ Compute anomalies with respect to a given reference period.

        Args:
            - data: dictionary, preprocessed data
            - lon_size, lat_size: longitude size and latitude size
            - nan_idx: list of integers, list of nan indices in the flatten array
            - time_period: Int, time series lentgh (target period 1981-2015)
            
        Return:
            - data_reshaped: dictionary of reshaped data (stacked across runs)
            - means, vars: dictionaries of means and variances
    """
    data_reshaped = {}
    means = {}
    vars = {}


    for idx_m,m in enumerate(data.keys()):
        data_reshaped[m] = np.zeros((len(data[m].keys()),time_period, lat_size*lon_size))
        
    
        for idx_r, r in enumerate(data[m].keys()):

            # flatten the data
            data_reshaped[m][idx_r,:,:] = data[m][r].copy().reshape(time_period, lat_size*lon_size)

            # replace continent's grid cell values with NaNs
            data_reshaped[m][idx_r,:,nan_idx] = float('nan')

        # compute the mean  ########
        means[m] = np.zeros_like(data_reshaped[m])
        # means[m] = np.nanmean(data_reshaped[m],axis=(0))
        # means[m] = np.expand_dims(means[m],axis=(0))
        # means[m] = np.repeat(means[m], data_reshaped[m].shape[0], axis=0)

        # compute the variance
        vars[m] = np.nanvar(data_reshaped[m],axis=(0))
        vars[m] = np.expand_dims(vars[m],axis=(0))
        vars[m] = np.repeat(vars[m], data_reshaped[m].shape[0], axis=0)

        # compute the anomalies
        data_reshaped[m] = data_reshaped[m] - np.expand_dims(np.nanmean(data_reshaped[m],axis=1),axis=1).repeat(data_reshaped[m].shape[1],axis=1)
        
    return data_reshaped, means, vars


def compute_smooth_variance(data, smoothing=0.1):
    """ Compute smoothed variance.

        Args:
            - data: dictionary, preprocessed data
            - time_period: Int, time series lentgh (target period 1981-2015)
            
        Return:
            - data_anomalies: dictionary of centered (according to a specific definition) data
    """
    # compute the forced response
    vars = {}

    for idx_m,m in enumerate(data.keys()):
        
        var_tmp= np.nanvar(data[m],axis=0)
        sigma0 = np.nanmedian(var_tmp,axis=1).expand_dims(axis=1).repeat(var_tmp.shape[1],axis=1)

        # compute the smoothed variance
        vars[m] = smoothing*sigma0 + (1-smoothing)*var_tmp

        print('Shape of the variance :',vars[m].shape)

    return vars




def compute_forced_response(data):
    """ Compute forced response.

        Args:
            - data: dictionary, preprocessed data
            - lon_size, lat_size: Integer, longitude size and latitude size
            - nan_idx: list of integers, list of nan indices in the flatten array
            - time_period: Int, time series lentgh (target period 1981-2015)
            
        Return:
            - data_anomalies: dictionary of centered (according to a specific definition) data
    """
    # compute the forced response
    data_forced_response = {}

    for idx_m,m in enumerate(list(data.keys())):
        
        data_forced_response[m] = np.expand_dims(np.nanmean(data[m],axis=0),axis=0).repeat(data[m].shape[0],axis=0)

    
    return data_forced_response



def merge_runs(x,y,means,vars):
    """ Merge runs for each model.

        Args:
            x: dictionary, anomalies (stacked)
            y: dictionary, forced response (stacked)
            vars: dictionary, variance (stacked)
            
        Return:
            x_merged, y_merged, vars_merged: dictionaries, concatenate runs for each model
    """
    y_merged = {}
    x_merged = {}
    means_merged = {}    
    vars_merged = {}

    
    for idx_m,m in enumerate(x.keys()):

        # get grid dimension
        d = x[m].shape[2]

        # concatenate across runs
        y_merged[m] = y[m].view(-1,d)
        x_merged[m] = x[m].view(-1,d)

        # concatenatye means across runs
        means_merged[m] = means[m].view(-1,d)

        # concatenate variance  across runs
        vars_merged[m] = vars[m].view(-1,d)
    
    return x_merged, y_merged, means_merged, vars_merged



def numpy_to_torch(x,y,means,vars, dtype=torch.float32):
    x_tmp = {}
    y_tmp = {}
    means_tmp = {}
    vars_tmp = {}

    for idx_m,m in enumerate(x.keys()):
        x_tmp[m] = torch.from_numpy(x[m]).to(dtype)
        y_tmp[m] = torch.from_numpy(y[m]).to(dtype)
        means_tmp[m] = torch.from_numpy(means[m]).to(dtype)
        vars_tmp[m] = torch.from_numpy(vars[m]).to(dtype)
        
    return x_tmp, y_tmp, means_tmp, vars_tmp


def merge_models_and_runs(m_out,x,y,means,vars,dtype=torch.float32):
    """Concatenate training sets for all models except model m. This enables to create the big matrices X and Y.
        Unnormalized data.
       Args:

       Return:
    """
    # merge runs for each model
    x_merged, y_merged, means_merged, vars_merged = merge_runs(x.copy(),y.copy(),means,vars)

    ################ We construct X, Y in R^{grid x runs*time steps}

    # We construct X_test in R^{grid x runs*time steps} using scaler computed in the training set
    x_test = None

    # We construct Y_test in R^{grid x runs*time steps} using TRUE scaler
    y_test = None

    # Concatenate all models to build the matrix X
    training_models = []
    count_tmp = 0
    
    for idx_m,m in enumerate(x.keys()):
        
        if m != m_out:
            training_models.append(m)
            if count_tmp ==0:

                x_train = x_merged[m]/np.sqrt(x_merged[m].shape[0])
                y_train = y_merged[m]/np.sqrt(x_merged[m].shape[0])
                count_tmp +=1

            else:
                x_train = torch.cat([x_train, x_merged[m]/ np.sqrt(x_merged[m].shape[0])],dim=0)
                y_train = torch.cat([y_train, y_merged[m]/ np.sqrt(x_merged[m].shape[0])],dim=0)

        else:
            # we do not add the model m_out to the training set
            x_test = x_merged[m]
            y_test = y_merged[m]
    return training_models, x_train, y_train, x_test, y_test


def rescale_training_set(m_out,x,y,means,vars,dtype=torch.float32):
    """Stack all ensemble members except for model m. This enables to create the big matrices X and Y.

       Args:

       Return:
    """
    # compute the test mean and variance mean as the mean of the variance for all training climate model.

    # compute dictionary of rescaled data
    x_rescaled = {}
    y_rescaled = {}

    # Concatenate all models to build the matrix X
    training_models = []
    count_tmp = 0
    
    for idx_m,m in enumerate(x.keys()):

        if m != m_out:
            x_rescaled[m] = (x[m] - means[m] )/ (torch.sqrt(vars[m]))
            y_rescaled[m] = (y[m] - means[m] )/ (torch.sqrt(vars[m]))
            training_models.append(m)
        
        else:
            x_rescaled[m] = x[m] 
            y_rescaled[m] = y[m]

    return training_models, x_rescaled, y_rescaled



# reshape the data such that X is (Time, Runs, lat*lon)
def stack_models_and_runs(models,x,y, dtype=torch.float32):
    """Stack all ensemble members of all models in a single tensor.

       Args:

       Return:
    """

    for idx_m,m in enumerate(models):
        if idx_m == 0:
            x_stacked = x[m]
            y_stacked = y[m]
        else:   
            x_stacked = torch.cat((x_stacked, x[m]), dim=0)
            y_stacked = torch.cat((y_stacked, y[m]), dim=0)

    return x_stacked, y_stacked


def merge_training_data(x_dict: Dict, y_dict: Dict, 
                       dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge training data from multiple models into single tensors.
    
    Args:
        x_dict: Dictionary of input data
        y_dict: Dictionary of target data
        dtype: PyTorch data type
        
    Returns:
        x_train: Merged input tensor
        y_train: Merged target tensor
    """
    for idx_m, model in enumerate(x_dict.keys()):
        model_x = torch.from_numpy(x_dict[model]).to(dtype)
        model_y = torch.from_numpy(y_dict[model]).to(dtype)
        
        # Reshape and normalize by sqrt of number of runs
        model_x = model_x.reshape(-1, model_x.shape[-2] * model_x.shape[-1])
        model_y = model_y.reshape(-1, model_y.shape[-2] * model_y.shape[-1])
        
        norm_factor = torch.sqrt(torch.tensor(x_dict[model].shape[0], dtype=dtype))
        model_x /= norm_factor
        model_y /= norm_factor
        
        if idx_m == 0:
            x_train = model_x
            y_train = model_y
        else:
            x_train = torch.cat([x_train, model_x], dim=0)
            y_train = torch.cat([y_train, model_y], dim=0)
            
    return x_train, y_train

def reshape_training_data(x_dict: Dict, y_dict: Dict,
                         dtype: torch.dtype = torch.float32) -> Tuple[Dict, Dict]:
    """
    Reshape training data while keeping model separation.

    Handles inputs with shapes:
      X/Y: (runs, years, lat, lon)
      X/Y: (runs, years, spatial)
    """
    x_train_dict = {}
    y_train_dict = {}

    for model in x_dict.keys():
        x_data = x_dict[model]
        y_data = y_dict[model]


        # Enforce expected dims
        if x_data.ndim == 4:  # (runs, years, lat, lon)
            xr = torch.from_numpy(x_data).to(dtype)
            yr = torch.from_numpy(y_data).to(dtype)
            xr = xr.view(x_data.shape[0], x_data.shape[1], -1)
            yr = yr.view(y_data.shape[0], y_data.shape[1], -1)
        elif x_data.ndim == 3:  # (runs, years, spatial)
            xr = torch.from_numpy(x_data).to(dtype)
            yr = torch.from_numpy(y_data).to(dtype)
        else:
            raise ValueError(f"Model {model}: unsupported x_data shape {x_data.shape}")

        if xr.shape[:2] != yr.shape[:2]:
            raise ValueError(
                f"Model {model}: mismatch in (runs, years) dims X {xr.shape[:2]} vs Y {yr.shape[:2]}"
            )

        if yr.shape[2] == 1 and xr.shape[2] > 1:
            # Broadcast singleton target feature across spatial dimension if intended
            yr = yr.expand(yr.shape[0], yr.shape[1], xr.shape[2])

        x_train_dict[model] = xr
        y_train_dict[model] = yr

    return x_train_dict, y_train_dict

def stack_models_and_runs(model_list: List[str], x_dict: Dict, y_dict: Dict,
                         dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stack all ensemble members of all models into single tensors.
    
    Args:
        model_list: List of model names
        x_dict: Dictionary of input data
        y_dict: Dictionary of target data
        dtype: PyTorch data type
        
    Returns:
        x_stacked: Stacked input tensor
        y_stacked: Stacked target tensor
    """
    for idx_m, model in enumerate(model_list):
        model_x = torch.from_numpy(x_dict[model]).view(x_dict[model].shape[0], x_dict[model].shape[1], -1)
        model_y = torch.from_numpy(y_dict[model]).view(y_dict[model].shape[0], y_dict[model].shape[1], -1)
        
        if idx_m == 0:
            x_stacked = model_x
            y_stacked = model_y
        else:
            x_stacked = torch.cat((x_stacked, model_x), dim=0)
            y_stacked = torch.cat((y_stacked, model_y), dim=0)
            
    return x_stacked.to(dtype), y_stacked.to(dtype)

# Smoothing functions
def moving_average_smoothing(x: torch.Tensor, window_size: int = 5, 
                           mode: str = 'same') -> torch.Tensor:
    """
    Apply moving average smoothing to multivariate time series.
    
    Args:
        x: Input tensor of shape (n_samples, n_timesteps, n_features)
        window_size: Size of moving average window
        mode: Padding mode ('same' or 'valid')
        
    Returns:
        Smoothed tensor
    """
    n_samples, n_timesteps, n_features = x.shape
    
    # Create uniform kernel
    kernel = torch.ones(1, 1, window_size, dtype=x.dtype, device=x.device) / window_size
    
    # Reshape for conv1d
    x_reshaped = x.permute(0, 2, 1).reshape(n_samples * n_features, 1, n_timesteps)
    
    if mode == 'same':
        padding = window_size // 2
        x_padded = F.pad(x_reshaped, (padding, padding), mode='reflect')
        smoothed = F.conv1d(x_padded, kernel)
        
        # Ensure same length as input
        if smoothed.shape[-1] > n_timesteps:
            smoothed = smoothed[:, :, :n_timesteps]
        elif smoothed.shape[-1] < n_timesteps:
            padding_needed = n_timesteps - smoothed.shape[-1]
            smoothed = F.pad(smoothed, (0, padding_needed), mode='replicate')
    else:
        smoothed = F.conv1d(x_reshaped, kernel)
    
    # Reshape back
    if mode == 'same':
        smoothed = smoothed.reshape(n_samples, n_features, n_timesteps).permute(0, 2, 1)
    else:
        new_timesteps = smoothed.shape[-1]
        smoothed = smoothed.reshape(n_samples, n_features, new_timesteps).permute(0, 2, 1)
    
    return smoothed

def exponential_smoothing(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    Apply exponential smoothing to time series.
    
    Args:
        x: Input tensor of shape (n_samples, n_timesteps, n_features)
        alpha: Smoothing parameter (0 < alpha <= 1)
        
    Returns:
        Smoothed tensor
    """
    smoothed = torch.zeros_like(x)
    smoothed[:, 0, :] = x[:, 0, :]
    
    for t in range(1, x.shape[1]):
        smoothed[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * smoothed[:, t-1, :]
    
    return smoothed

def gaussian_smoothing(x: torch.Tensor, window_size: int = 5, 
                      sigma: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian smoothing to time series.
    
    Args:
        x: Input tensor of shape (n_samples, n_timesteps, n_features)
        window_size: Size of Gaussian kernel (should be odd)
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Smoothed tensor
    """
    n_samples, n_timesteps, n_features = x.shape
    
    # Create Gaussian kernel
    kernel_range = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    kernel = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, window_size)
    
    # Reshape for conv1d
    x_reshaped = x.permute(0, 2, 1).reshape(n_samples * n_features, 1, n_timesteps)
    
    # Apply padding and convolution
    padding = window_size // 2
    x_padded = F.pad(x_reshaped, (padding, padding), mode='reflect')
    smoothed = F.conv1d(x_padded, kernel)
    
    # Ensure same length as input
    if smoothed.shape[-1] > n_timesteps:
        smoothed = smoothed[:, :, :n_timesteps]
    elif smoothed.shape[-1] < n_timesteps:
        padding_needed = n_timesteps - smoothed.shape[-1]
        smoothed = F.pad(smoothed, (0, padding_needed), mode='replicate')
    
    # Reshape back
    smoothed = smoothed.reshape(n_samples, n_features, n_timesteps).permute(0, 2, 1)
    
    return smoothed



def filter_training_data_by_indices(x_train_merged: torch.Tensor, 
                                   y_train_merged: torch.Tensor,
                                   valid_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter merged training data to keep only valid spatial indices.
    
    Args:
        x_train_merged: Input training data tensor (n_samples, n_features)
        y_train_merged: Target training data tensor (n_samples, n_features)
        valid_indices: List of valid spatial indices to keep
        
    Returns:
        Tuple of (filtered_x_train, filtered_y_train)
    """
    x_train_filtered = x_train_merged[:, valid_indices]
    y_train_filtered = y_train_merged[:, valid_indices]
    
    return x_train_filtered, y_train_filtered


def capture_nans(x_train_dict):
    """
    Capture indices of NaN values across all training data.
    Args:
        x_train_dict (dict): Dictionary of training data tensors with keys as model names.
    Returns:
        nan_union (list): List of indices where NaN values are found across all models.
        notnan_inter (list): List of indices where no NaN values are found across all models.
    """
    
    # enumerate in the dictionary and get the union of the nans for each pair ((key, value))
    for (key, value) in x_train_dict.items():
        # nan_indices = list(torch.where(torch.abs(value[0,:,:]) > 1e9)[0].numpy())

        # new code to test 
        # get nan mask of test set 
        nan_mask = np.where(np.abs(value[0,:,:])>1e10, True, False)

        if nan_mask.any() == False:
            nan_mask = np.where(np.isnan(value[0,:,:])==True, True, False)

        # get the index of columns where there is at least one True in the nan mask
        col_indices = np.where(np.any(nan_mask, axis=0))[0]

        if 'nan_union' not in locals():
            nan_union = col_indices
        else:
            nan_union = list(set(nan_union) | set(col_indices))

    notnan_inter = list(set(range(value.shape[2])) - set(nan_union))

    return nan_union, notnan_inter


def filter_training_dict_by_indices(x_train_dict: Dict[str, torch.Tensor],
                                   y_train_dict: Dict[str, torch.Tensor],
                                   valid_indices: List[int],
                                   verbose: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Filter training data dictionaries to keep only valid spatial indices.
    
    Args:
        x_train_dict: Dictionary of input training tensors
        y_train_dict: Dictionary of target training tensors
        valid_indices: List of valid spatial indices to keep
        verbose: Whether to print progress
        
    Returns:
        Tuple of (filtered_x_dict, filtered_y_dict)
    """
    x_train_filtered = {}
    y_train_filtered = {}
    
    if verbose:
        print(f"Filtering training dictionaries...")
        print(f"Keeping {len(valid_indices)} valid spatial features")
    
    for model_name in x_train_dict.keys():
        # Filter input data (shape: n_samples, n_time, n_spatial)
        x_train_filtered[model_name] = x_train_dict[model_name][:, :, valid_indices]
        
        # Filter target data (shape: n_samples, n_time, n_spatial)
        print(y_train_dict[model_name].shape)
        y_train_filtered[model_name] = y_train_dict[model_name][:, :, valid_indices]
        
        if verbose:
            original_shape_x = x_train_dict[model_name].shape
            filtered_shape_x = x_train_filtered[model_name].shape
            print(f"{model_name}: X {original_shape_x} -> {filtered_shape_x}")
    
    return x_train_filtered, y_train_filtered

def filter_test_data_by_indices(x_test: torch.Tensor,
                               valid_indices: List[int]) -> torch.Tensor:
    """
    Filter test data to keep only valid spatial indices.
    
    Args:
        x_test: Test data tensor (n_samples, n_time, n_spatial)
        valid_indices: List of valid spatial indices to keep
        
    Returns:
        Filtered test data tensor
    """
    return x_test[:, :, valid_indices]

def apply_notnan_filter_complete(x_train_merged: torch.Tensor,
                                y_train_merged: torch.Tensor,
                                x_train_dict: Dict[str, torch.Tensor],
                                y_train_dict: Dict[str, torch.Tensor],
                                x_test: torch.Tensor,
                                notnan_indices: List[int],
                                verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, 
                                                              Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                                                              torch.Tensor]:
    """
    Apply not-NaN filter to all training and test data consistently.
    
    Args:
        x_train_merged: Merged input training data
        y_train_merged: Merged target training data
        x_train_dict: Dictionary of input training data
        y_train_dict: Dictionary of target training data
        x_test: Test data
        notnan_indices: List of valid (not-NaN) spatial indices
        verbose: Whether to print progress
        
    Returns:
        Tuple of (filtered_x_train_merged, filtered_y_train_merged,
                 filtered_x_train_dict, filtered_y_train_dict, filtered_x_test)
    """
    if verbose:
        print("="*70)
        print("              APPLYING NOT-NAN FILTER TO ALL DATA")
        print("="*70)
        print(f"Valid indices count: {len(notnan_indices)}")
    
    # Filter merged training data
    if verbose:
        print("\n1. Filtering merged training data...")
    x_train_merged_filtered, y_train_merged_filtered = filter_training_data_by_indices(
        x_train_merged, y_train_merged, notnan_indices
    )
    
    if verbose:
        print(f"   X: {x_train_merged.shape} -> {x_train_merged_filtered.shape}")
        print(f"   Y: {y_train_merged.shape} -> {y_train_merged_filtered.shape}")
    
    # Filter training dictionaries
    if verbose:
        print("\n2. Filtering training dictionaries...")
    x_train_dict_filtered, y_train_dict_filtered = filter_training_dict_by_indices(
        x_train_dict, y_train_dict, notnan_indices, verbose=verbose
    )
    
    # Filter test data
    if verbose:
        print(f"\n3. Filtering test data...")
    x_test_filtered = filter_test_data_by_indices(x_test, notnan_indices)
    
    if verbose:
        print(f"   Test: {x_test.shape} -> {x_test_filtered.shape}")
        
        # Summary
        original_features = x_train_merged.shape[1]
        filtered_features = len(notnan_indices)
        reduction_pct = (1 - filtered_features/original_features) * 100
        
        print(f"\n=== FILTERING SUMMARY ===")
        print(f"Original spatial features: {original_features}")
        print(f"Filtered spatial features: {filtered_features}")
        print(f"Features removed: {original_features - filtered_features}")
        print(f"Reduction: {reduction_pct:.1f}%")
    
    return (x_train_merged_filtered, y_train_merged_filtered,
            x_train_dict_filtered, y_train_dict_filtered, x_test_filtered)
