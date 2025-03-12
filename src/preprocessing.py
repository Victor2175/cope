# Tool functions to preprocessed the data from pkl dictionary to centered and standardized data

# Upscaling/Downscaling library
import skimage
import numpy as np
import torch


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
        
        if (len(data[m].keys()) > 3) and (idx_m < max_models):

            data_processed[m] = data[m].copy()
            
            for idx_r, r in enumerate(data[m].keys()):

                # Upscaling of raw data 
                data_processed[m][r] = skimage.transform.downscale_local_mean(data_processed[m][r][131:,:,:],(1,2,2))
                data_processed[m][r] = data_processed[m][r][:,latitude<=60,:]

                # capture nan indices and record the union of nans
                nan_idx_tmp = list(np.where(np.isnan(data_processed[m][r][0,:,:].ravel())==True)[0])
                nan_idx = list(set(nan_idx) | set(nan_idx_tmp))

    # get longitude and latitude size
    lon_size = longitude.shape[0]
    lat_size = latitude[latitude <=60].shape[0]
    
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

            # center the data with respect to the given mean across time
            # data_reshaped[m][idx_r,:,:] = data_reshaped[m][idx_r,:,:]

        # compute the mean 
        means[m] = np.nanmean(data_reshaped[m],axis=0)

        # compute the variance
        vars[m] = np.nanvar(data_reshaped[m],axis=0)
        
    return data_reshaped, means, vars


def compute_forced_response(data, lon_size, lat_size, nan_idx, time_period=34):
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

    for idx_m,m in enumerate(data.keys()):
        
        data_forced_response[m] = np.zeros((len(data[m].keys()),time_period, lat_size*lon_size))
        y_tmp = np.zeros((len(data[m].keys()),time_period, lat_size*lon_size))
    
        for idx_r, r in enumerate(data[m].keys()):

            # flatten the data
            y_tmp[idx_r,:,:] = data[m][r].copy().reshape(time_period, lat_size*lon_size)

            # replace continent's grid cell values with NaNs
            y_tmp[idx_r,:,nan_idx] = float('nan')
    
        # compute mean reference
        mean_spatial_ensemble = np.nanmean(y_tmp,axis=0)

        # copmpute forced response (the same for each run)
        for idx_r, r in enumerate(data[m].keys()):              
            data_forced_response[m][idx_r,:,:] = mean_spatial_ensemble

    return data_forced_response



def merge_runs(x,y,vars):
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
    vars_merged = {}
    
    for idx_m,m in enumerate(x.keys()):

        # get grid dimension
        d = x[m].shape[2]

        # concatenate across runs
        y_merged[m] = y[m].view(-1,d)
        x_merged[m] = x[m].view(-1,d)

        # concatenate variance  across runs
        vars_merged[m] = vars[m]
    
    return x_merged, y_merged, vars_merged



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



def rescale_and_merge_training_and_test_sets(m_out,x,y,means,vars,dtype=torch.float32):
    """Concatenate training sets for all models except model m. This enables to create the big matrices X and Y.
        The data are standardized as follow: 
        training data are divided by the square root of the variance for each climate model.
        test data are divided by the mean of the square root of the variance for all training climate model.

       Args:

       Return:
    """
    # merge runs for each model
    x_merged, y_merged, vars_merged = merge_runs(x,y,vars)

    # compute mean (training scaler as mean across climate models)
    means_mean_merged = torch.mean(torch.stack([torch.mean(means[m],axis=0,dtype=dtype) for m in x.keys() if m != m_out]),axis=0,dtype=dtype)
    
    # compute variance (training scaler as mean across climate models)
    vars_mean_merged_tmp = torch.mean(torch.stack([vars[m] for m in x.keys() if m != m_out]),axis=0,dtype=dtype)

    # duplicate the variance for each run of the test model 
    d = x[m_out].shape[2]
    time_period = x[m_out].shape[1]

    # merge variance for each run of the test model
    vars_mean_merged = vars_mean_merged_tmp.repeat(x[m_out].shape[0],1)

    ################ We construct X, Y in R^{grid x runs*time steps}

    # We construct X_test in R^{grid x runs*time steps} using scaler computed in the training set
    x_test = (x_merged[m_out] - means_mean_merged)/torch.sqrt(vars_mean_merged)

    # We construct Y_test in R^{grid x runs*time steps} using TRUE scaler
    y_test = (y_merged[m_out] - torch.mean(means[m_out],axis=0,dtype=dtype))/torch.sqrt(vars_mean_merged)

    # Concatenate all models to build the matrix X
    training_models = []
    count_tmp = 0
    
    for idx_m,m in enumerate(x.keys()):
        
        if m != m_out:
            training_models.append(m)
            if count_tmp ==0:

                x_train = (x_merged[m]  - torch.mean(means[m],axis=0,dtype=dtype))/torch.sqrt(vars[m].repeat(x[m].shape[0],1))
                y_train = (y_merged[m] - torch.mean(means[m],axis=0,dtype=dtype))/torch.sqrt(vars[m].repeat(x[m].shape[0],1))
                count_tmp +=1

            else:
                x_train = torch.cat([x_train, (x_merged[m] - torch.mean(means[m],axis=0,dtype=dtype))/torch.sqrt(vars[m].repeat(x[m].shape[0],1)) ],dim=0)
                y_train = torch.cat([y_train, (y_merged[m] - torch.mean(means[m],axis=0,dtype=dtype))/torch.sqrt(vars[m].repeat(x[m].shape[0],1)) ],dim=0)

    return training_models, x_train, y_train, x_test, y_test


def rescale_training_and_test_sets(m_out,x,y,means,vars,dtype=torch.float32):
    """Stack all ensemble members except for model m. This enables to create the big matrices X and Y.

       Args:

       Return:
    """
    # compute the test mean and variance mean as the mean of the variance for all training climate model.
    # means_mean = torch.mean(torch.stack([means[m] for m in x.keys() if m != m_out]),axis=0, dtype=dtype)
    means_mean = torch.mean(torch.stack([torch.mean(means[m],axis=0) for m in x.keys() if m != m_out]),axis=0, dtype=dtype)
    vars_mean = torch.mean(torch.stack([vars[m] for m in x.keys() if m != m_out]),axis=0, dtype=dtype)

    # compute dictionary of rescaled data
    x_rescaled = {}
    y_rescaled = {}

    # Concatenate all models to build the matrix X
    training_models = []
    count_tmp = 0
    
    for idx_m,m in enumerate(x.keys()):

        if m != m_out:
            x_rescaled[m] = (x[m] - torch.mean(means[m],axis=0, dtype=dtype))/torch.sqrt(vars[m])
            y_rescaled[m] = (y[m] - torch.mean(means[m],axis=0, dtype=dtype))/torch.sqrt(vars[m])
            training_models.append(m)
        
        else:

            x_rescaled[m] = (x[m] - means_mean)/torch.sqrt(vars_mean)
            y_rescaled[m] = (y[m] - torch.mean(means[m],axis=0, dtype=dtype))/torch.sqrt(vars[m])

    return training_models, x_rescaled, y_rescaled

def stack_runs_for_each_model(models,x,y):
    """Stack all ensemble members for each model. This enables to create the big matrices X and Y.

       Args:

       Return:
    """
    # compute dictionary of rescaled data
    x_rescaled = {}
    y_rescaled = {}

    for idx_m,m in enumerate(models):
        x_rescaled[m] = x[m].view(-1,x[m].shape[0])
        y_rescaled[m] = y[m].view(-1,x[m].shape[0])

    return x_rescaled, y_rescaled