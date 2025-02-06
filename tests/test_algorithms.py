import unittest
import torch # type: ignore
import numpy as np
from src.algorithms import ridge_regression, ridge_regression_low_rank, low_rank_projection, compute_gradient, train_robust_weights_model, compute_weights, prediction

class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        # Set up common test data
        self.X = torch.randn(10, 25, dtype=torch.float32)
        self.Y = torch.randn(10, 25, dtype=torch.float32)
        self.models = ['model1', 'model2']
        self.x = {m: torch.randn(10, 5, 25, dtype=torch.float32) for m in self.models}
        self.y = {m: torch.randn(10, 5, 25, dtype=torch.float32) for m in self.models}
        self.w = torch.randn(25, 25, dtype=torch.float32)
        self.notnan_idx = list(np.arange(25))
        self.nan_idx = []
        self.lon_size = 5
        self.lat_size = 5

    def test_ridge_regression(self):
        W_ols = ridge_regression(self.X, self.Y, lambda_=1.0)
        self.assertEqual(W_ols.shape, (25, 25))

    def test_ridge_regression_low_rank(self):
        W_rrr = ridge_regression_low_rank(self.X, self.Y, rank=2, lambda_=1.0)
        self.assertEqual(W_rrr.shape, (25, 25))

    def test_low_rank_projection(self):
        M = torch.randn(10, 10, dtype=torch.float32)
        M_low_rank = low_rank_projection(M, rank=3)
        self.assertEqual(M_low_rank.shape, (10, 10))

    def test_compute_gradient(self):
        grad = compute_gradient(self.models, self.x, self.y, self.w, self.notnan_idx, lambda_=1.0, mu_=1.0)
        self.assertEqual(grad.shape, (25, 25))

    def test_train_robust_weights_model(self):
        w = train_robust_weights_model(self.models, self.x, self.y, self.lon_size, self.lat_size, self.notnan_idx, rank=2, lambda_=1.0, mu_=1.0, nb_iterations=5)
        self.assertEqual(w.shape, (25, 25))

    def test_compute_weights(self):
        weights = compute_weights(self.models, self.w, self.x, self.y, self.notnan_idx, lambda_=1.0, mu_=1.0)
        self.assertEqual(len(weights), len(self.models))

    def test_prediction(self):
        x = torch.randn(10, 25, dtype=torch.float32)
        W = torch.randn(25, 25, dtype=torch.float32)
        y_pred = prediction(x, W, self.notnan_idx, self.nan_idx)
        self.assertEqual(y_pred.shape, (10, 25))

if __name__ == '__main__':
    unittest.main()