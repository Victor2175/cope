# Tool functions to preprocessed the data from pkl dictionary to centered and standardized data

# Upscaling/Downscaling library
import skimage
import numpy as np
import torch
import netCDF4 as netcdf


def data_processing(data,longitude,latitude,max_nb_models = 10):
    """ Process the data: statically relevant climate models (nb_runs > 3),
                            upscaling (from 0.25 to 0.5), 
                            cropping (remove latitude > 60),
                            time period focus (from 1981 until 2014)

        Args: 
            data: Dictionary of raw data (indexed by models and subdictionary indexed by runs)
            longitude, latitude: np.array, longitude and latitude coordinates.
            
        
        Returns:
            data_processed: upscaled data
            notnan_idx, nan_idx: non nan indices and nan indices i
    """
    # first filter out the models that contain less than 3 ensemble members 
    data_processed = {}

    # record nan indices as the union of nans on each map
    nan_idx = []


    for idx_m,m in enumerate(data.keys()):
        
        if (len(data[m].keys()) > 3) and (idx_m < max_nb_models):

            data_processed[m] = data[m].copy()
            
            for idx_r, r in enumerate(data[m].keys()):

                # Upscaling of raw data 
                data_processed[m][r] = skimage.transform.downscale_local_mean(data_processed[m][r][131:164,:,:],(1,2,2))
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


def extract_longitude_latitude(file = '/net/h2o/climphys3/simondi/cope-analysis/data/erss/sst_annual_g050_mean_19812014_centered.nc'):
    """
    Extract longitude and latitude coordinates as np.array from netcdf file. 
    
    """
    # read the dataset
    file2read = netcdf.Dataset(file,'r')
    
    # load longitude, latitude and sst monthly means
    lon = np.array(file2read.variables['lon'][:])
    lat = np.array(file2read.variables['lat'][:])
    sst = np.array(file2read.variables['sst'])
    
    # define grid (+ croping for latitude > 60)
    lat_grid, lon_grid = np.meshgrid(lat[lat<=60], lon, indexing='ij')
    
    lat_size = lat_grid.shape[0]
    lon_size = lon_grid.shape[1]

    return lon, lat, lon_grid, lat_grid


def compute_anomalies(data, lon_size, lat_size, nan_idx, time_period=33):
    """ Compute anomalies with respect to a given reference period.

        Args:
            - data: dictionary, preprocessed data
            - lon_size, lat_size: longitude size and latitude size
            - nan_idx: list of integers, list of nan indices in the flatten array
            - time_period: Int, time series lentgh (target period 1981-2014)
            
        Return:
            - data_anomalies: dictionary of centered (according to a specific definition) data
    """
    data_anomalies = {}

    for idx_m,m in enumerate(data.keys()):
        data_anomalies[m] = data[m].copy()

        y_tmp = np.zeros((len(data[m].keys()),time_period, lat_size*lon_size))
    
        for idx_r, r in enumerate(data[m].keys()):

            # flatten the data
            y_tmp[idx_r,:,:] = data[m][r].copy().reshape(time_period, lat_size*lon_size)

            # replace continent's grid cell values with NaNs
            y_tmp[idx_r,:,nan_idx] = float('nan')
            

        # compute mean reference
        mean_ref_ensemble = np.nanmean(y_tmp,axis=1)
        mean_ref_ensemble = np.nanmean(mean_ref_ensemble,axis=0)

        # center the data with respect to the given mean
        for idx_r, r in enumerate(data[m].keys()):
            
            # compute anomalies
            data_anomalies[m][r] = y_tmp[idx_r,:,:] - mean_ref_ensemble

    return data_anomalies


def compute_forced_response(data, lon_size, lat_size, nan_idx, time_period=33):
    """ Compute forced response.

        Args:
            - data: dictionary, preprocessed data
            - lon_size, lat_size: Integer, longitude size and latitude size
            - nan_idx: list of integers, list of nan indices in the flatten array
            - time_period: Int, time series lentgh (target period 1981-2014)
            
        Return:
            - data_anomalies: dictionary of centered (according to a specific definition) data
    """
    # compute the forced response
    data_forced_response = {}

    for idx_m,m in enumerate(data.keys()):
        
        data_forced_response[m] = data[m].copy()

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
            data_forced_response[m][r] = mean_spatial_ensemble - np.nanmean(mean_spatial_ensemble,axis=0)

    return data_forced_response


def compute_variance(data,lon_size, lat_size, nan_idx, time_period=33):
    """ Compute forced response.

        Args:
            - data: dictionary, preprocessed data
            - lon_size, lat_size: Integer, longitude size and latitude size
            - nan_idx: list of integers, list of nan indices in the flatten array
            - time_period: Int, time series lentgh (target period 1981-2014)
            
        Return:
            - data_anomalies: dictionary of centered (according to a specific definition) data
    """

    # compute the variance
    variance = {}

    for idx_m,m in enumerate(data.keys()):
        
        variance[m] = {}
        arr_tmp = np.zeros((len(data[m].keys()),time_period,lon_size*lat_size))
        
        for idx_r, r in enumerate(list(data[m].keys())):
            arr_tmp[idx_r,:,:] = data[m][r]
    
        variance[m] = np.var(arr_tmp,axis=0)

    return variance

def merge_runs(x,y,vars):
    """ Merge runs for each model.

        Args:
            x: dictionary, anomalies
            y: dictionary, forced response
            vars: dictionary, variance
            
        Return:
            x_merged, y_merged, vars_merged: dictionaries, concatenated runs for eadch model
    """
    y_merged = {}
    x_merged = {}
    vars_merged = {}
    
    for idx_m,m in enumerate(x.keys()):
        y_merged[m] = 0
        x_merged[m] = 0
        vars_merged[m] = 0
    
        for idx_r, r in enumerate(x[m].keys()):

            if idx_r ==0:
                y_merged[m] = y[m][r]
                x_merged[m] = x[m][r]
                vars_merged[m] = vars[m]
            else:
                y_merged[m] = torch.cat([y_merged[m],y[m][r]],dim=0 )
                x_merged[m] = torch.cat([x_merged[m], x[m][r]],dim=0)  
                vars_merged[m] = torch.cat([vars_merged[m], vars[m]],dim=0)
    
    return x_merged, y_merged, vars_merged    

def numpy_to_torch(x,y,vars):
    x_tmp = {}
    y_tmp = {}
    vars_tmp = {}

    for idx_m,m in enumerate(x.keys()):
        x_tmp[m] = {}
        y_tmp[m] = {}
        vars_tmp[m] = {}
        
        for idx_r, r in enumerate(x[m].keys()):
            x_tmp[m][r] = torch.from_numpy(x[m][r]).to(torch.float64)
            y_tmp[m][r] = torch.from_numpy(y[m][r]).to(torch.float64)
            
        vars_tmp[m] = torch.from_numpy(vars[m]).to(torch.float64)

    return x_tmp, y_tmp, vars_tmp


def standardize(x,y,vars,merged=False):
    """ Standardize dataset. 

        Args:
            x: dictionary, anomalies
            y: dictionary, forced response
            vars: dictionary, variance

        Return:
            x_tmp, y_tmp: dictionary, 
    """
    x_tmp = {}
    y_tmp = {}
    vars_tmp = {}
    
    for idx_m,m in enumerate(x.keys()):
        x_tmp[m] = {}
        y_tmp[m] = {}
        vars_tmp[m] = {}

        if merged == False:
            for idx_r, r in enumerate(x[m].keys()):
                x_tmp[m][r] = x[m][r]/torch.sqrt(vars[m])
                y_tmp[m][r] = y[m][r]/torch.sqrt(vars[m])
        else:
            x_tmp[m] = x[m]/torch.sqrt(vars[m])
            y_tmp[m] = y[m]/torch.sqrt(vars[m])

    
    return x_tmp, y_tmp


def build_training_and_test_sets(m_out,x,y,vars,lon_size,lat_size,time_period=33):
    """Concatenate training sets for all models except model m. This enables to create the big matrices X and Y.

       Args:

       Return:
    """
    # merge runs for each model
    x_merged, y_merged, vars_merged = merge_runs(x,y,vars)

    # We construct X, Y in R^{grid x runs*time steps}
    x_train = 0
    y_train = 0

    # We construct X_test, Y_test in R^{grid x runs*time steps}
    x_test = x_merged[m_out]
    y_test = y_merged[m_out]

    # Concatenate all models to build the matrix X
    training_models = []
    count_tmp = 0
    
    for idx_m,m in enumerate(x.keys()):

        if m != m_out:
            training_models.append(m)
            if count_tmp ==0:
                x_train = x_merged[m]
                y_train = y_merged[m]
                count_tmp +=1 
                
            else:
                x_train = torch.cat([x_train, x_merged[m]],dim=0)
                y_train = torch.cat([y_train, y_merged[m]],dim=0)

    return training_models, x_train, y_train, x_test, y_test