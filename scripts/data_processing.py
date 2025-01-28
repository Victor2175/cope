# This script extract data from (/net/atmos/data/cmip6-ng/tos/ann/g025) to build SST time series of CMIP6 models

# import libraries
import numpy as np
import matplotlib
import netCDF4 as netcdf
import os
import sys
import pickle

#############################################
# Get the list of all files and directories
PATH = "/net/atmos/data/cmip6-ng/tos/ann/g025"
DIR_LIST = os.listdir(PATH)

print("Files and directories in '", PATH, "' :")


####### Initialize the directories #############
dic_files = {}

list_model = []
list_forcing = []

for idx, file in enumerate(DIR_LIST):

    file_split = file.split("_")
    
    # extract model names
    model_name = file_split[2]
    forcing = file_split[3]
    run_name = file_split[4]
    
    
    list_model.append(model_name)
    list_forcing.append(forcing)
    
model_names = list(set(list_model))
forcing_names = list(set(list_forcing))

########## Initialize dictionaries #######www 
dic_model_forcing = {}
for idx,model in enumerate(model_names):
    dic_model_forcing[model] = {}
    for forcing in forcing_names:
        dic_model_forcing[model][forcing] = {}

#########  Extract the data from each file (model, forcing, ensemble member) ###############


for idx, file in enumerate(DIR_LIST):

    file_split = file.split("_")

    # extract model names
    model = file_split[2]
    forcing = file_split[3]
    run_name = file_split[4]
    
    
    if model in list(dic_model_forcing.keys()):
          
        # read files in the directory
        file2read = netcdf.Dataset(PATH +'/'+ file,'r')

        # set variables
        time = np.array(file2read.variables['time'][:])
        longitude = np.array(file2read.variables['lon'][:])
        latitude = np.array(file2read.variables['lat'][:])
        tos = np.array(file2read.variables['tos'][:])
        
        if (forcing == 'historical'):
            print(file)
      
        # assign nans to non-sea values
        tos[tos>1e19] = np.nan
        idx_nans = np.argwhere(np.isnan(tos))

        # get the data
        dic_model_forcing[model][forcing][run_name]= tos



###########  Extract the useful data: we keep historical data ################

# define forcing 
forcing_hist = "historical"
dic_runs_hist = {i: [] for i in model_names}

for idx, model in enumerate(model_names):
    
    dic_runs_hist[model] = {}

    for idx_key, key in enumerate(dic_model_forcing[model][forcing_hist].keys()):
        
        # load the runs
        dic_runs_hist[model][key] = dic_model_forcing[model][forcing_hist][key]


############# Write pickle file that contains historical data ################
# create a binary pickle file 
f = open("../data/ssp585_time_series.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(dic_runs_hist,f)

# close file
f.close()