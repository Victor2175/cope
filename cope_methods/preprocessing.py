"""
Preprocessing utilities: temporal aggregation, reshaping, NaN filtering, smoothing.
"""

from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import torch


# ----------------------
# Temporal aggregation
# ----------------------
def yearly_average(data: np.ndarray, months_per_year: int = 12) -> np.ndarray:
    """
    Compute yearly average from monthly data.

    Args:
        data: (..., time, *spatial)
    """
    n_years = data.shape[1] // months_per_year
    trimmed = data[:, : n_years * months_per_year]
    reshaped = trimmed.reshape(
        data.shape[0], n_years, months_per_year, *data.shape[2:]
    )
    return np.nanmean(reshaped, axis=2)


def compute_yearly_average_dict(
    data_dict: Dict[str, np.ndarray], months_per_year: int = 12
) -> Dict[str, np.ndarray]:
    return {k: yearly_average(v, months_per_year) for k, v in data_dict.items()}


# ----------------------
# Reshaping for ML
# ----------------------
def merge_training_data(
    dic_data_yearly: Dict[str, np.ndarray],
    dic_forced_yearly: Dict[str, np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge all per-model yearly ensembles into single arrays.

    Returns:
        X: (total_members, years, features)
        Y: (total_members, years, features) -> repeated forced response per member
    """
    X_list, Y_list = []
    for key in dic_data_yearly:
        dd = dic_data_yearly[key]  # (members, years, lat, lon)
        forced = dic_forced_yearly[key]  # (years, lat, lon)
        members, years = dd.shape[:2]
        feats = np.prod(dd.shape[2:])
        X_list.append(dd.reshape(members, years, feats))
        forced_flat = forced.reshape(years, feats)
        Y_list.append(np.repeat(forced_flat[None, ...], members, axis=0))
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return (
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(Y.astype(np.float32)),
    )


def reshape_training_data(
    dic_data_yearly: Dict[str, np.ndarray],
    dic_forced_yearly: Dict[str, np.ndarray],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Keep per-model structure.
    """
    Xd, Yd = {}, {}
    for k in dic_data_yearly:
        dd = dic_data_yearly[k]
        forced = dic_forced_yearly[k]
        members, years = dd.shape[:2]
        feats = np.prod(dd.shape[2:])
        Xd[k] = torch.from_numpy(dd.reshape(members, years, feats).astype(np.float32))
        Yd[k] = torch.from_numpy(forced.reshape(years, feats).astype(np.float32))
    return Xd, Yd


# ----------------------
# Missing values
# ----------------------
def capture_nans(
    x_train_dict: Dict[str, torch.Tensor]
) -> Tuple[List[int], List[int]]:
    """
    Identify feature indices (flattened spatial) that contain any NaN.
    """
    nan_features = set()
    first_shape = next(iter(x_train_dict.values())).shape[2]
    for tensor in x_train_dict.values():
        arr = tensor.numpy()
        mask = np.any(np.isnan(arr), axis=(0, 1))
        nan_features |= set(np.where(mask)[0])
    all_features = set(range(first_shape))
    not_nan = list(all_features - nan_features)
    return list(nan_features), not_nan


def apply_notnan_filter_complete(
    x_merge: torch.Tensor,
    y_merge: torch.Tensor,
    x_dict: Dict[str, torch.Tensor],
    y_dict: Dict[str, torch.Tensor],
    x_test: torch.Tensor,
    keep_idx: List[int],
):
    """
    Slice feature dimension across all aligned data structures.
    """
    x_merge_f = x_merge[:, :, keep_idx]
    y_merge_f = y_merge[:, :, keep_idx]
    x_dict_f = {k: v[:, :, keep_idx] for k, v in x_dict.items()}
    y_dict_f = {k: v[:, keep_idx] for k, v in y_dict.items()}
    x_test_f = x_test[:, :, keep_idx]
    return x_merge_f, y_merge_f, x_dict_f, y_dict_f, x_test_f


# ----------------------
# Smoothing
# ----------------------
def moving_average_smoothing(
    data: torch.Tensor, window_size: int = 60, mode: str = "same"
) -> torch.Tensor:
    """
    Temporal moving average along time axis.

    Args:
        data: (runs, time, features)
        mode: 'same' pads to preserve length
    """
    if window_size < 2:
        return data.clone()
    pad = window_size // 2 if mode == "same" else 0
    kernel = torch.ones(window_size, device=data.device) / window_size
    padded = torch.nn.functional.pad(data, (0, 0, pad, pad), mode="replicate")
    out = torch.zeros_like(data)
    for t in range(data.shape[1]):
        window = padded[:, t : t + window_size]  # (runs, window, feats)
        out[:, t] = torch.tensordot(window, kernel, dims=([1], [0]))
    return out