"""
Visualization helpers (Robinson projection + comparison).
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False


class ForceSMIPVisualizer:
    def __init__(self, longitude: np.ndarray, latitude: np.ndarray):
        self.lon = longitude
        self.lat = latitude
        if not _HAS_CARTOPY:
            print("[visualization] Cartopy not installed; map plots disabled.")

    # ---------------
    def _to_grid(self, vec: np.ndarray) -> np.ndarray:
        return vec.reshape(self.lat.shape[0], self.lon.shape[0])

    # ---------------
    def plot_robinson_projection(
        self,
        data: np.ndarray,
        title: str,
        run_idx: int = 0,
        cmap: str = "RdBu_r",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        central_longitude: float = 180,
    ):
        if not _HAS_CARTOPY:
            raise RuntimeError("Cartopy not available.")
        grid = self._to_grid(data[run_idx])
        if vmin is None or vmax is None:
            vmax = np.nanmax(np.abs(grid))
            vmin = -vmax
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111, projection=ccrs.Robinson(central_longitude=central_longitude))
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        im = ax.pcolormesh(
            lon2d,
            lat2d,
            grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(title)
        cb = plt.colorbar(im, orientation="horizontal", pad=0.05, shrink=0.8)
        cb.set_label("Trend (units / year)")
        return fig

    # ---------------
    def plot_triple_comparison(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        member: np.ndarray,
        run_idx: int = 0,
        vmin: float = -0.05,
        vmax: float = 0.05,
        cmap: str = "RdBu_r",
    ):
        if not _HAS_CARTOPY:
            raise RuntimeError("Cartopy not available.")
        fig = plt.figure(figsize=(15, 4))
        datasets = [
            (ground_truth, "Ground Truth"),
            (prediction, "Prediction"),
            (member, "Test Member"),
        ]
        lon2d, lat2d = np.meshgrid(self.lon, self.lat)
        for i, (arr, title) in enumerate(datasets):
            ax = plt.subplot(1, 3, i + 1, projection=ccrs.Robinson(central_longitude=180))
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            grid = self._to_grid(arr[run_idx])
            im = ax.pcolormesh(
                lon2d,
                lat2d,
                grid,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            ax.set_title(title, fontsize=10)
        cb = plt.colorbar(im, ax=fig.axes, orientation="horizontal", pad=0.05, shrink=0.85)
        cb.set_label("Trend (units / year)")
        return fig

    # ---------------
    @staticmethod
    def plot_performance_comparison(
        comparison_results: Dict[str, dict],
        metric: str = "mean_nrmse",
        title: str = "Method Comparison",
        rotation: int = 45,
    ):
        methods = list(comparison_results.keys())
        values = [comparison_results[m][metric] for m in methods]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(methods, values, color="#4C72B0")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticklabels(methods, rotation=rotation, ha="right")
        ax.grid(axis="y", alpha=0.3)
        return fig