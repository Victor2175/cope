"""
Metric and statistical utilities.
"""

from __future__ import annotations
import numpy as np
from typing import Dict


def compute_trends_from_data(
    data_yearly: np.ndarray, years_slice: slice
) -> np.ndarray:
    """
    Linear trend via least-squares slope over selected years.

    Args:
        data_yearly: (runs, years, features)
    """
    sub = data_yearly[:, years_slice, :]
    t = np.arange(sub.shape[1], dtype=float)
    t_center = t - t.mean()
    denom = np.sum(t_center**2)
    slopes = np.sum(sub * t_center[None, :, None], axis=1) / denom
    return slopes


def pattern_correlation(a: np.ndarray, b: np.ndarray) -> float:
    am = a - np.nanmean(a)
    bm = b - np.nanmean(b)
    num = np.nansum(am * bm)
    den = np.sqrt(np.nansum(am**2) * np.nansum(bm**2)) + 1e-12
    return float(num / den)


def amplitude_ratio(pred: np.ndarray, tgt: np.ndarray) -> float:
    return float((np.nanstd(pred) + 1e-12) / (np.nanstd(tgt) + 1e-12))


def compute_statistics(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p75": float(np.nanpercentile(arr, 75)),
    }