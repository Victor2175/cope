import os
import netCDF4 as netcdf
import numpy as np
from typing import Dict, Tuple, List, Optional

VARIABLE_MAP = {
    'tasmax': ('monmaxtasmax', 'tasmax'),
    'tasmin': ('monmintasmin', 'tasmin'),
    'prmax': ('monmaxpr', 'pr'),
    'zmta': ('zmta', 'ta'),
    'tas': ('tas', 'tas'),
    'tos': ('tos', 'tos'),
    'pr': ('pr', 'pr'),
    'psl': ('psl', 'psl')
    # Add more mappings if needed
}

class ForceSMIPDataLoader:
    """Class to handle loading and preprocessing of ForceSMIP data."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        
    def _map_variable(self, variable):
        """Helper to map variable names for file and netCDF variable."""
        if variable in VARIABLE_MAP:
            variable_tmp, variable_nc = VARIABLE_MAP[variable]
        else:
            variable_tmp, variable_nc = variable, variable
        return variable_tmp, variable_nc
    
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
        variable_tmp, variable_nc = self._map_variable(variable)

        # Fix the path construction - add 'ForceSMIP' subdirectory
        path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/Amon/{variable_tmp}')
        
        if not os.path.exists(path):
            path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/Omon/{variable_tmp}')

        if not os.path.exists(path):
            path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/Lmon/{variable_tmp}')

        if not os.path.exists(path):
            path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/OImon/{variable_tmp}')

        if not os.path.exists(path):
            path = os.path.join(self.base_path, 'ForceSMIP', f'Training-Ext/Aday/{variable_tmp}')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data path not found: {path}")
            
        dir_list = os.listdir(path)
        
        dic_data = {}
        dic_forced_response = {}
        
        for idx_m, model_dir in enumerate(dir_list):
            print(f'Processing {idx_m+1}/{len(dir_list)}: {model_dir}')
            dir_path = os.path.join(path, model_dir)
            file_list = os.listdir(dir_path)
            
            if variable_tmp != 'zmta':
                dic_data[model_dir] = np.zeros((len(file_list), 2652, 72, 144), dtype=np.float32)
                dic_forced_response[model_dir] = np.zeros((len(file_list), 2652, 72, 144), dtype=np.float32)
            else:
                dic_data[model_dir] = np.zeros((len(file_list), 2652, 17, 72), dtype=np.float32)
                dic_forced_response[model_dir] = np.zeros((len(file_list), 2652, 17, 72), dtype=np.float32)

            for idx_f, file in enumerate(file_list):
                print(f'  Processing {idx_f+1}/{len(file_list)}: {file}')
                file_path = os.path.join(dir_path, file)
                
                with netcdf.Dataset(file_path, 'r') as nc_file:
                    time = np.array(nc_file.variables['time'])
                    longitude = np.array(nc_file.variables['lon'])
                    latitude = np.array(nc_file.variables['lat'])
                    data = np.array(nc_file.variables[variable_nc])


                    # Monthly centering
                    dic_data[model_dir][idx_f, :, :, :] = data.squeeze()
                    data_tmp = data.copy()
                    data_tmp[np.abs(data_tmp) > 1e9] = np.nan  #

                    for i in range(12):
                        month_mask = np.arange(time.shape[0]) % 12 == i
                        # monthly_mean = np.nanmean(dic_data[model_dir][idx_f, month_mask, :, :], axis=0)
                        
                        monthly_mean = np.nanmean(data_tmp[month_mask, :, :], axis=0).squeeze()
                        dic_data[model_dir][idx_f, month_mask, :, :] -= monthly_mean
                    
            # Compute forced response as ensemble mean
            dic_forced_response[model_dir][:,:,:,:] = np.nanmean(dic_data[model_dir], axis=0)

        return dic_data, dic_forced_response, longitude, latitude

    def load_test_data(self, variable: str = 'tas', 
                      tier: str = 'Tier1',
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
            test_models = ['1B', '1D', '1E', '1G', '1J']

        variable_tmp, variable_nc = self._map_variable(variable)

        # Fix the path construction
        path = os.path.join(self.base_path, f'ForceSMIP/Evaluation-{tier}/')
        
        # Fix the path construction - add 'ForceSMIP' subdirectory
        path_tmp = os.path.join(path, f'Amon/{variable_tmp}')
        
        if not os.path.exists(path_tmp):
            path_tmp = os.path.join(path, f'Omon/{variable_tmp}')

        if not os.path.exists(path_tmp):
            path_tmp = os.path.join(path, f'Aday/{variable_tmp}')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Test data path not found: {path}")


        file_list = os.listdir(path_tmp)
        print(f"Files in test data directory: {file_list}")

        start = 1

        for idx_file, file in enumerate(file_list):
            if file.startswith(f'{variable_tmp}_') and file.endswith('.nc'):
                for idx_model, model in enumerate(test_models):
                    if model in file:
                        idx_test = test_models.index(model)
                        print(f'Loading test data for {model} at index {idx_test}')

                        file_path = os.path.join(path_tmp, file)
                        with netcdf.Dataset(file_path, 'r') as nc_file:
                            time = np.array(nc_file.variables['time'])
                            data = np.array(nc_file.variables[variable_nc])
                            if start == 1:
                                start = 0
                                print("create data_test array")
                                data_test = np.zeros((len(test_models), data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
                                data_test[idx_model, :, :, :] = data.squeeze()
                            else:
                                data_test[idx_model, :, :, :] = data.squeeze()

                        data_tmp = data_test[idx_model, :, :, :].copy()
                        data_tmp[np.abs(data_tmp) > 1e9] = np.nan

                        print("Monthly centering")
                        # Monthly centering
                        for i in range(12):
                            month_mask = np.arange(time.shape[0]) % 12 == i
                            monthly_mean = np.nanmean(data_tmp[month_mask, :, :], axis=0).squeeze()
                            # monthly_mean = np.nanmean(data_test[idx_model, :, :, :], axis=0).squeeze()
                            data_test[idx_model, month_mask, :, :] -= monthly_mean
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
            test_models = ['1B', '1D', '1E', '1G', '1J']

        variable_tmp, variable_nc = self._map_variable(variable)

        # Fix the path construction
        path = os.path.join(self.base_path, 'ForceSMIP_Tier1_final/ensmeans-Tier1')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth path not found: {path}")
            
        file_list = os.listdir(path)
        
        if variable != 'zmta':
            data_ground_truth = np.zeros((len(test_models), 876, 72, 144), dtype=np.float32)
        else:
            data_ground_truth = np.zeros((len(test_models), 876, 17, 72), dtype=np.float32)

        for file in file_list:
            if f'.{variable_tmp}.' in file:
                for idx_model, model in enumerate(test_models):
                    if model in file:
                        print(f'Loading ground truth for {model} at index {idx_model}')

                        file_path = os.path.join(path, file)
                        with netcdf.Dataset(file_path, 'r') as nc_file:
                            time = np.array(nc_file.variables['time'])
                            data = np.array(nc_file.variables['arr_EM'])

                            data_ground_truth[idx_model, :, :, :] = data.squeeze()
                            

                        data_tmp = data_ground_truth[idx_model, :, :, :].copy()
                        data_tmp[np.abs(data_tmp) > 1e9] = np.nan

                        # Monthly centering
                        for i in range(12):
                            month_mask = np.arange(time.shape[0]) % 12 == i
                            # monthly_mean = np.nanmean(data_ground_truth[idx_model, month_mask, :, :], axis=0).squeeze()
                            monthly_mean = np.nanmean(data_tmp[month_mask, :, :], axis=0).squeeze()
                            data_ground_truth[idx_model, month_mask, :, :] -= monthly_mean
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
            test_models = ['1B', '1D', '1E', '1G', '1J']
        if methods_to_center is None:
            methods_to_center = [8, 9, 14, 24]

        variable_tmp, variable_nc = self._map_variable(variable)

        # Fix the path construction
        path = os.path.join(self.base_path, 'ForceSMIP_Tier1_final/ForceSMIP-estimates-Tier1')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Estimates path not found: {path}")
            
        file_list = os.listdir(path)
        
        if variable != 'zmta':
            data_estimates = np.zeros((30, len(test_models), 876, 72, 144), dtype=np.float32)
        else:
            data_estimates = np.zeros((22, len(test_models), 876, 17, 72), dtype=np.float32)

        for file in file_list:
            if file.startswith(f'{variable_tmp}_'):
                file_path = os.path.join(path, file)
                
                with netcdf.Dataset(file_path, 'r') as nc_file:
                    time = np.array(nc_file.variables['time'])
                    
                    for idx_model, model in enumerate(test_models):
                        if model in file:
                            data_estimates[:, idx_model, :, :, :] = np.array(nc_file.variables['forced_component']).squeeze()

        # Center specific methods
        for idx_m in methods_to_center:
            if idx_m >= data_estimates.shape[0]:
                print(f"Method index {idx_m} exceeds data shape, skipping centering.")
                continue
            print(f'Centering data for method {idx_m}')
            for i in range(12):
                month_mask = np.arange(data_estimates.shape[2]) % 12 == i
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
    return np.nanmean(data_reshaped, axis=2)

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