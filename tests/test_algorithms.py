import sys, os
import numpy as np
import torch
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ForceSMIP"))

from algorithms import ridge_regression, LowRankSolver

@pytest.mark.parametrize("feat_out", [20, 15])
def test_ridge_regression_shapes(feat_out):
    runs, T, F = 12, 30, 20
    X = torch.randn(runs, T, F)
    W_true = torch.randn(F, feat_out)
    Y = (X @ W_true) + 0.05 * torch.randn(runs, T, feat_out)
    print(X.shape, Y.shape, W_true.shape)
    W = ridge_regression(X, Y, lambda_reg=10.0)
    assert W.shape == (F, feat_out)

def test_low_rank_solver():
    runs, T, F = 10, 24, 16
    X = torch.randn(runs, T, F)
    Y = torch.randn(runs, T, F)
    solver = LowRankSolver()
    solver.fit(X, Y, lambda_reg=50.0)
    Wk = solver.get_rank_k_solution(rank=5)
    assert Wk.shape == (F, F)
    cvs = solver.cumulative_variance_ratio()
    assert torch.all(cvs >= 0) and cvs[-1] <= 1.00001