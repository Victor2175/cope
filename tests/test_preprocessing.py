import unittest
import numpy as np
import torch

from src.preprocessing import data_processing, compute_anomalies, compute_forced_response, compute_variance, stack_runs, merge_runs, numpy_to_torch, standardize, build_training_and_test_sets

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Set up common test data
        self.data = {
            'model1': {0: np.random.rand(200, 144, 72), 1: np.random.rand(200, 144, 72), 2: np.random.rand(200, 144, 72), 3: np.random.rand(200, 144, 72)},
            'model2': {0: np.random.rand(200, 144, 72), 1: np.random.rand(200, 144, 72), 2: np.random.rand(200, 144, 72), 3: np.random.rand(200, 144, 72)},
        }
        self.longitude = np.linspace(0, 360, 72)
        self.latitude = np.linspace(-90, 90, 144)
        self.lon_size = 72
        self.lat_size = 30
        self.time_period = 33
        self.nan_idx = [0, 1, 2]
        self.notnan_idx = list(set(range(self.lon_size * self.lat_size)) - set(self.nan_idx))

    def test_data_processing(self):
        data_processed, notnan_idx, nan_idx = data_processing(self.data, self.longitude, self.latitude)
        self.assertTrue(isinstance(data_processed, dict))
        self.assertTrue(isinstance(notnan_idx, list))
        self.assertTrue(isinstance(nan_idx, list))

    def test_compute_anomalies(self):
        data_anomalies = compute_anomalies(self.data, self.lon_size, self.lat_size, self.nan_idx, self.time_period)
        self.assertTrue(isinstance(data_anomalies, dict))

    def test_compute_forced_response(self):
        data_forced_response = compute_forced_response(self.data, self.lon_size, self.lat_size, self.nan_idx, self.time_period)
        self.assertTrue(isinstance(data_forced_response, dict))

    def test_compute_variance(self):
        variance = compute_variance(self.data, self.lon_size, self.lat_size, self.nan_idx, self.time_period)
        self.assertTrue(isinstance(variance, dict))

    def test_numpy_to_torch(self):
        x_torch, y_torch, vars_torch = numpy_to_torch(self.data, self.data, self.data)
        self.assertTrue(isinstance(x_torch, dict))
        self.assertTrue(isinstance(y_torch, dict))
        self.assertTrue(isinstance(vars_torch, dict))

    def test_stack_runs(self):
        data_tmp_x, data_tmp_y, data_tmp_vars = numpy_to_torch(self.data,self.data,self.data)
        x_stacked, y_stacked, vars_stacked = stack_runs(data_tmp_x, data_tmp_y, data_tmp_vars, self.time_period, self.lon_size, self.lat_size)
        self.assertTrue(isinstance(x_stacked, dict))
        self.assertTrue(isinstance(y_stacked, dict))
        self.assertTrue(isinstance(vars_stacked, dict))

    def test_merge_runs(self):
        data_tmp_x, data_tmp_y, data_tmp_vars = numpy_to_torch(self.data,self.data,self.data)
        x_stacked, y_stacked, vars_stacked = stack_runs(data_tmp_x, data_tmp_y, data_tmp_vars, self.time_period, self.lon_size, self.lat_size)
        x_merged, y_merged, vars_merged = merge_runs(x_stacked, y_stacked, vars_stacked)
        self.assertTrue(isinstance(x_merged, dict))
        self.assertTrue(isinstance(y_merged, dict))
        self.assertTrue(isinstance(vars_merged, dict))

    def test_standardize(self):
        x_torch, y_torch, vars_torch = numpy_to_torch(self.data, self.data, self.data)
        x_standardized, y_standardized = standardize(x_torch, y_torch, vars_torch)
        self.assertTrue(isinstance(x_standardized, dict))
        self.assertTrue(isinstance(y_standardized, dict))

    def test_build_training_and_test_sets(self):
        data_tmp_x, data_tmp_y, data_tmp_vars = numpy_to_torch(self.data,self.data,self.data)
        x_stacked, y_stacked, vars_stacked = stack_runs(data_tmp_x, data_tmp_y, data_tmp_vars, self.time_period, self.lon_size, self.lat_size)
        training_models, x_train, y_train, x_test, y_test = build_training_and_test_sets('model1', x_stacked, y_stacked, vars_stacked, self.lon_size, self.lat_size, self.time_period)
        self.assertTrue(isinstance(training_models, list))
        self.assertTrue(isinstance(x_train, torch.Tensor))
        self.assertTrue(isinstance(y_train, torch.Tensor))
        self.assertTrue(isinstance(x_test, torch.Tensor))
        self.assertTrue(isinstance(y_test, torch.Tensor))

if __name__ == '__main__':
    unittest.main()