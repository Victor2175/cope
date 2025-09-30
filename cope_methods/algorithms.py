"""
Core modeling algorithms: ridge regression, low-rank solver, per-model weighted ridge.
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional
import torch
import numpy as np


# ----------------------
# Ridge Regression
# ----------------------
def ridge_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_reg: float,
    surface: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Solve W in (X^T X + λI) W = X^T Y (row-wise across runs; time averaged).

    Assumes:
        x: (runs, T, F)
        y: (runs, T, F_out) OR (runs, T, F)  (forced response replicated)

    Returns:
        W: (F, F_out)
    """
    # Collapse time by mean; alternative weighting can be added here
    X_mean = torch.nanmean(x, dim=1)  # (runs, F)
    Y_mean = torch.nanmean(y, dim=1)  # (runs, F_out)

    if surface is not None:
        # Weighted Gram
        # surface expected shape (F,)
        Wsurf = torch.diag(surface)
        XTX = X_mean.T @ (Wsurf @ X_mean)
        XTY = X_mean.T @ (Wsurf @ Y_mean)
    else:
        XTX = X_mean.T @ X_mean
        XTY = X_mean.T @ Y_mean

    I = torch.eye(XTX.shape[0], device=XTX.device, dtype=XTX.dtype)
    A = XTX + lambda_reg * I
    W = torch.linalg.solve(A, XTY)

    if verbose:
        print(f"[ridge] solved weights shape={W.shape}")
    return W


# ----------------------
# Low-Rank Approximation Wrapper
# ----------------------
class LowRankSolver:
    """
    Stores SVD of full ridge solution to cheaply extract rank-k approximations.
    """

    def __init__(self):
        self.W_full: Optional[torch.Tensor] = None
        self.U: Optional[torch.Tensor] = None
        self.S: Optional[torch.Tensor] = None
        self.Vt: Optional[torch.Tensor] = None
        self.is_fitted: bool = False

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_reg: float,
        surface: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ):
        self.W_full = ridge_regression(x, y, lambda_reg, surface, verbose=False)
        U, S, Vt = torch.linalg.svd(self.W_full, full_matrices=False)
        self.U, self.S, self.Vt = U, S, Vt
        self.is_fitted = True
        if verbose:
            print(f"[LowRankSolver] SVD rank={S.numel()}")
        return self

    def get_rank_k_solution(
        self, rank: int, verbose: bool = False
    ) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        k = min(rank, self.S.shape[0])
        if verbose and k < rank:
            print(f"[LowRankSolver] rank clipped to {k}")
        S_k = torch.diag(self.S[:k])
        return self.U[:, :k] @ S_k @ self.Vt[:k, :]

    def get_multiple_ranks(self, ranks: Iterable[int], verbose: bool = False):
        return {r: self.get_rank_k_solution(r, verbose=verbose) for r in ranks}

    def cumulative_variance_ratio(self) -> torch.Tensor:
        sv2 = self.S**2
        return torch.cumsum(sv2, dim=0) / torch.sum(sv2)


# ----------------------
# Per-model Weighted Ridge
# ----------------------
class WeightedRidgeRegression:
    """
    Train separate ridge (and low-rank) per model; combine with learned or uniform weights.
    """

    def __init__(self):
        self.full_weights: Dict[str, torch.Tensor] = {}
        self.low_rank_weights: Dict[str, torch.Tensor] = {}

    def train_per_model(
        self,
        x_dict,
        y_dict,
        lambda_reg: float,
        rank: int,
        verbose: bool = False,
    ):
        performance = {}
        for key in x_dict:
            W_full = ridge_regression(x_dict[key], y_dict[key].unsqueeze(1), lambda_reg)
            lr = LowRankSolver().fit(x_dict[key], y_dict[key].unsqueeze(1), lambda_reg)
            W_rank = lr.get_rank_k_solution(rank)
            self.full_weights[key] = W_full
            self.low_rank_weights[key] = W_rank

            # Simple training error proxy
            pred_full = torch.nanmean(x_dict[key], dim=1) @ W_full
            pred_lr = torch.nanmean(x_dict[key], dim=1) @ W_rank
            mse_full = torch.nanmean((pred_full - y_dict[key]) ** 2).item()
            mse_lr = torch.nanmean((pred_lr - y_dict[key]) ** 2).item()
            performance[key] = {"mse_full": mse_full, "mse_low_rank": mse_lr}
            if verbose:
                print(f"[WeightedRidge] {key}: full={mse_full:.4e} rank={mse_lr:.4e}")
        return performance

    def compute_weights(
        self,
        x_dict,
        y_dict,
        use_low_rank: bool = False,
        uniform: bool = True,
    ) -> torch.Tensor:
        keys = list(x_dict.keys())
        if uniform:
            return torch.ones(len(keys)) / len(keys)
        errs = []
        for k in keys:
            W = self.low_rank_weights[k] if use_low_rank else self.full_weights[k]
            pred = torch.nanmean(x_dict[k], dim=1) @ W
            err = torch.nanmean((pred - y_dict[k]) ** 2).item()
            errs.append(err)
        errs = np.asarray(errs)
        w = 1.0 / (errs + 1e-8)
        return torch.from_numpy(w / w.sum()).float()

    def predict_weighted(
        self,
        x_test: torch.Tensor,
        weights: torch.Tensor,
        use_low_rank: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x_test: (runs, time, features)
            weights: (n_models,)
        """
        preds = []
        for i, (k, W) in enumerate(
            (self.low_rank_weights.items() if use_low_rank else self.full_weights.items())
        ):
            preds.append((torch.nanmean(x_test, dim=1) @ W)[None, ...])
        stack = torch.cat(preds, dim=0)  # (n_models, runs, features_out)
        return torch.tensordot(weights, stack, dims=([0], [0]))