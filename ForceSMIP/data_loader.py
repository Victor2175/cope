import os
import netCDF4 as netcdf
import numpy as np
from typing import Dict, Tuple, List, Optional

class ForceSMIPDataLoader:
    """Class to handle loading and preprocessing of ForceSMIP data."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        
    def load_training_data(self, variable: str = 'tas') -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
        """
        Load training data from ForceSMIP dataset.
        
        Args:
            variable: Variable name ('tas' or 'tos')
            
        Returns:
            dic_data: Dictionary of raw data indexed by model
            dic_forced_response: Dictionary of forced response data
            longitude: Longitude coordinates
            latitude: Latitude coordinates
        """
        # Fix the path construction - add 'ForceSMIP' subdirectory
        path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/Amon/{variable}')
        
        if not os.path.exists(path):
            path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/Omon/{variable}')

        if not os.path.exists(path):
            path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/Lmon/{variable}')

        if not os.path.exists(path):
            path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/OImon/{variable}')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data path not found: {path}")
            
        dir_list = os.listdir(path)
        
        dic_data = {}
        dic_forced_response = {}
        
        for idx_m, model_dir in enumerate(dir_list):
            print(f'Processing {idx_m+1}/{len(dir_list)}: {model_dir}')
            dir_path = os.path.join(path, model_dir)
            file_list = os.listdir(dir_path)
            
            dic_data[model_dir] = np.zeros((len(file_list), 2652, 72, 144), dtype=np.float32)
            dic_forced_response[model_dir] = np.zeros((len(file_list), 2652, 72, 144), dtype=np.float32)
            
            for idx_f, file in enumerate(file_list):
                print(f'  Processing {idx_f+1}/{len(file_list)}: {file}')
                file_path = os.path.join(dir_path, file)
                
                with netcdf.Dataset(file_path, 'r') as nc_file:
                    time = np.array(nc_file.variables['time'][:])
                    longitude = np.array(nc_file.variables['lon'][:])
                    latitude = np.array(nc_file.variables['lat'][:])
                    data = np.array(nc_file.variables[variable])
                    
                    # Monthly centering
                    dic_data[model_dir][idx_f, :, :, :] = data
                    for i in range(12):
                        month_mask = np.arange(time.shape[0]) % 12 == i
                        monthly_mean = np.nanmean(dic_data[model_dir][idx_f, month_mask, :, :], axis=0)
                        dic_data[model_dir][idx_f, month_mask, :, :] -= monthly_mean
            
            # Compute forced response as ensemble mean
            dic_forced_response[model_dir][:, :, :, :] = np.nanmean(dic_data[model_dir], axis=0)
            
        return dic_data, dic_forced_response, longitude, latitude
    
    def load_test_data(self, variable: str = 'tas', 
                      test_models: Optional[List[str]] = None) -> np.ndarray:
        """
        Load test data from Evaluation-Tier1.
        
        Args:
            variable: Variable name ('tas' or 'tos')
            test_models: List of test model identifiers
            
        Returns:
            data_test: Test data array
        """
        if test_models is None:
            test_models = ['B', 'D', 'E', 'G', 'J']
            
        # Fix the path construction
        path = os.path.join(self.base_path, 'ForceSMIP_Tier1_final/Evaluation-Tier1/')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test data path not found: {path}")
            
        file_list = os.listdir(path)
        
        data_test = np.zeros((len(test_models), 876, 72, 144), dtype=np.float32)
        
        for file in file_list:
            if file.startswith(f'{variable}_') and file.endswith('.nc'):
                if file[9] in test_models:
                    idx_test = test_models.index(file[9])
                    print(f'Loading test data for {file[9]} at index {idx_test}')
                    
                    file_path = os.path.join(path, file)
                    with netcdf.Dataset(file_path, 'r') as nc_file:
                        time = np.array(nc_file.variables['time'])
                        data = np.array(nc_file.variables[variable])
                        data_test[idx_test, :, :, :] = data
                        
                        # Monthly centering
                        for i in range(12):
                            month_mask = np.arange(time.shape[0]) % 12 == i
                            monthly_mean = np.nanmean(data_test[idx_test, month_mask, :, :], axis=0)
                            data_test[idx_test, month_mask, :, :] -= monthly_mean
                            
        return data_test
    
    def load_ground_truth(self, variable: str = 'tas',
                         test_models: Optional[List[str]] = None) -> np.ndarray:
        """
        Load ground truth ensemble means.
        
        Args:
            variable: Variable name ('tas' or 'tos')
            test_models: List of test model identifiers
            
        Returns:
            data_ground_truth: Ground truth data array
        """
        if test_models is None:
            test_models = ['B', 'D', 'E', 'G', 'J']
            
        # Fix the path construction
        path = os.path.join(self.base_path, 'ForceSMIP_Tier1_final/ensmeans-Tier1')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth path not found: {path}")
            
        file_list = os.listdir(path)
        
        data_ground_truth = np.zeros((len(test_models), 876, 72, 144), dtype=np.float32)
        
        for file in file_list:
            if f'.{variable}.' in file:
                if file[1] in test_models:
                    idx_test = test_models.index(file[1])
                    print(f'Loading ground truth for {file[1]} at index {idx_test}')
                    
                    file_path = os.path.join(path, file)
                    with netcdf.Dataset(file_path, 'r') as nc_file:
                        time = np.array(nc_file.variables['time'])
                        data = np.array(nc_file.variables['arr_EM'])
                        data_ground_truth[idx_test, :, :, :] = data
                        
                        # Monthly centering
                        for i in range(12):
                            month_mask = np.arange(time.shape[0]) % 12 == i
                            monthly_mean = np.nanmean(data_ground_truth[idx_test, month_mask, :, :], axis=0)
                            data_ground_truth[idx_test, month_mask, :, :] -= monthly_mean
                            
        return data_ground_truth
    
    def load_estimates(self, variable: str = 'tas',
                      test_models: Optional[List[str]] = None,
                      methods_to_center: Optional[List[int]] = None) -> np.ndarray:
        """
        Load estimates from other methods.
        
        Args:
            variable: Variable name ('tas' or 'tos')
            test_models: List of test model identifiers
            methods_to_center: List of method indices that need centering
            
        Returns:
            data_estimates: Estimates data array
        """
        if test_models is None:
            test_models = ['B', 'D', 'E', 'G', 'J']
        if methods_to_center is None:
            methods_to_center = [8, 9, 14, 24]
            
        # Fix the path construction
        path = os.path.join(self.base_path, 'ForceSMIP_Tier1_final/ForceSMIP-estimates-Tier1')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Estimates path not found: {path}")
            
        file_list = os.listdir(path)
        
        data_estimates = np.zeros((30, len(test_models), 876, 72, 144), dtype=np.float32)
        
        for file in file_list:
            if file.startswith(f'{variable}_'):
                file_path = os.path.join(path, file)
                
                with netcdf.Dataset(file_path, 'r') as nc_file:
                    time = np.array(nc_file.variables['time'])
                    
                    if file[5] in test_models:
                        idx_test = test_models.index(file[5])
                        data_estimates[:, idx_test, :, :, :] = np.array(nc_file.variables['forced_component'])
        
        # Center specific methods
        for idx_m in methods_to_center:
            print(f'Centering data for method {idx_m}')
            for i in range(12):
                month_mask = np.arange(time.shape[0]) % 12 == i
                monthly_mean = np.nanmean(data_estimates[idx_m, :, month_mask, :, :], axis=0)
                data_estimates[idx_m, :, month_mask, :, :] -= monthly_mean
                
        return data_estimates

def yearly_average(data: np.ndarray, months_per_year: int = 12) -> np.ndarray:
    """
    Compute yearly average of time series data.
    
    Args:
        data: Input data array with time dimension
        months_per_year: Number of months per year
        
    Returns:
        Yearly averaged data
    """
    n_years = data.shape[1] // months_per_year
    data_reshaped = data.reshape(data.shape[0], n_years, months_per_year, *data.shape[2:])
    return data_reshaped.mean(axis=2)

def compute_yearly_average_dict(data_dict: Dict, months_per_year: int = 12) -> Dict:
    """
    Compute yearly average for dictionary of data arrays.
    
    Args:
        data_dict: Dictionary of data arrays
        months_per_year: Number of months per year
        
    Returns:
        Dictionary with yearly averaged data
    """
    yearly_data = {}
    for key, data in data_dict.items():
        yearly_data[key] = yearly_average(data, months_per_year)
    return yearly_data