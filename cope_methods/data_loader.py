"""
Data loading utilities for ForceSMIP datasets.

Responsibilities:
- Locate appropriate directories for training / evaluation / ground truth / estimates.
- Load NetCDF files and apply monthly climatology centering (remove seasonal cycle).
- Return structured arrays ready for downstream processing.

All monthly-centering is done per ensemble member before aggregation.
"""

from __future__ import annotations
import os
from typing import Dict, Tuple, List, Optional
import numpy as np
import netCDF4 as nc

from .constants import VARIABLE_MAP, DEFAULT_TEST_MODELS


class ForceSMIPDataLoader:
    """
    Loader for ForceSMIP project data.

    Attributes:
        base_path: Root path that contains ForceSMIP / ForceSMIP_Tier1_final directories.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path

    # ----------------------
    # Internal helpers
    # ----------------------
    def _map_variable(self, variable: str) -> Tuple[str, str]:
        """Return (folder_variable_name, netcdf_variable_name)."""
        return VARIABLE_MAP.get(variable, (variable, variable))

    def _find_first_existing(self, candidates: List[str]) -> str:
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No valid path among: {candidates}")

    # ----------------------
    # Training data
    # ----------------------
    def load_training_data(
        self,
        variable: str = "tas"
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Load extended training ensemble data and compute per-model forced response (ensemble mean).

        Returns:
            dic_data: {model: (n_members, time, lat, lon)}
            dic_forced_response: {model: (time, lat, lon)}
            longitude, latitude
        """
        var_dir, var_nc = self._map_variable(variable)
        root = os.path.join(self.base_path, "ForceSMIP", "Training-Ext")
        search_subdirs = ["Amon", "Omon", "Lmon", "OImon", "Aday"]
        path = self._find_first_existing([os.path.join(root, sd, var_dir) for sd in search_subdirs])

        dic_data: Dict[str, np.ndarray] = {}
        dic_forced: Dict[str, np.ndarray] = {}
        longitude = latitude = None

        models = sorted(os.listdir(path))
        for model in models:
            mdir = os.path.join(path, model)
            files = sorted(os.listdir(mdir))
            if var_dir != "zmta":
                shape = (len(files), 2652, 72, 144)
            else:
                shape = (len(files), 2652, 17, 72)
            dic_data[model] = np.zeros(shape, dtype=np.float32)

            for i, fname in enumerate(files):
                fpath = os.path.join(mdir, fname)
                with nc.Dataset(fpath) as ds:
                    time = np.asarray(ds.variables["time"])
                    longitude = np.asarray(ds.variables["lon"])
                    latitude = np.asarray(ds.variables["lat"])
                    arr = np.asarray(ds.variables[var_nc]).squeeze()
                    dic_data[model][i] = arr
                    arr_clean = arr.astype(float)
                    arr_clean[np.abs(arr_clean) > 1e9] = np.nan
                    # Remove monthly climatology
                    for m in range(12):
                        mask = (np.arange(time.size) % 12) == m
                        monthly_mean = np.nanmean(arr_clean[mask], axis=0)
                        dic_data[model][i, mask] -= monthly_mean

            dic_forced[model] = np.nanmean(dic_data[model], axis=0)

        if longitude is None or latitude is None:
            raise RuntimeError("Failed to read coordinate variables.")
        return dic_data, dic_forced, longitude, latitude

    # ----------------------
    # Test data
    # ----------------------
    def load_test_data(
        self,
        variable: str = "tas",
        tier: str = "Tier1",
        test_models: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load evaluation (test) members. Applies monthly-centering.

        Returns:
            data_test: (n_models, time, lat, lon)
            temporal_means: (n_models, time, lat, lon) repeated monthly climatology
        """
        if test_models is None:
            test_models = DEFAULT_TEST_MODELS
        var_dir, var_nc = self._map_variable(variable)
        root = os.path.join(self.base_path, f"ForceSMIP/Evaluation-{tier}")
        search = [os.path.join(root, sd, var_dir) for sd in ["Amon", "Omon", "Aday"]]
        path = self._find_first_existing(search)

        data_test = temporal_means = None

        for fname in os.listdir(path):
            if not (fname.startswith(f"{var_dir}_") and fname.endswith(".nc")):
                continue
            for midx, model in enumerate(test_models):
                if model not in fname:
                    continue
                fpath = os.path.join(path, fname)
                with nc.Dataset(fpath) as ds:
                    time = np.asarray(ds.variables["time"])
                    arr = np.asarray(ds.variables[var_nc]).squeeze()
                    if data_test is None:
                        data_test = np.zeros((len(test_models),) + arr.shape, dtype=np.float32)
                        temporal_means = np.zeros_like(data_test)
                    data_test[midx] = arr
                    arr2 = arr.astype(float)
                    arr2[np.abs(arr2) > 1e9] = np.nan
                    for m in range(12):
                        mask = (np.arange(time.size) % 12) == m
                        mmean = np.nanmean(arr2[mask], axis=0)
                        temporal_means[midx, mask] = mmean
                        data_test[midx, mask] -= mmean
        if data_test is None:
            raise FileNotFoundError("No test data loaded.")
        return data_test, temporal_means

    # ----------------------
    # Ground truth
    # ----------------------
    def load_ground_truth(
        self,
        variable: str = "tas",
        test_models: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Load ensemble mean (forced component) for test models.

        Returns:
            (n_models, time, lat, lon)
        """
        if test_models is None:
            test_models = DEFAULT_TEST_MODELS
        var_dir, _ = self._map_variable(variable)
        path = os.path.join(self.base_path, "ForceSMIP_Tier1_final", "ensmeans-Tier1")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        if variable != "zmta":
            shape = (len(test_models), 876, 72, 144)
        else:
            shape = (len(test_models), 876, 17, 72)
        out = np.zeros(shape, dtype=np.float32)

        for fname in os.listdir(path):
            if f".{var_dir}." not in fname:
                continue
            for midx, model in enumerate(test_models):
                if model not in fname:
                    continue
                with nc.Dataset(os.path.join(path, fname)) as ds:
                    time = np.asarray(ds.variables["time"])
                    arr = np.asarray(ds.variables["arr_EM"]).squeeze()
                    out[midx] = arr
                    arr2 = arr.astype(float)
                    arr2[np.abs(arr2) > 1e9] = np.nan
                    for m in range(12):
                        mask = (np.arange(time.size) % 12) == m
                        mmean = np.nanmean(arr2[mask], axis=0)
                        out[midx, mask] -= mmean
        return out

    # ----------------------
    # Third-party estimates
    # ----------------------
    def load_estimates(
        self,
        variable: str = "tas",
        test_models: Optional[List[str]] = None,
        methods_to_center: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Load pre-computed forced response estimates from other methods.

        Args:
            methods_to_center: indices (method axis) for which to re-center monthly means.

        Returns:
            data_estimates: (n_methods, n_models, time, lat, lon)
        """
        if test_models is None:
            test_models = DEFAULT_TEST_MODELS
        if methods_to_center is None:
            methods_to_center = [8, 9, 14, 24]

        var_dir, _ = self._map_variable(variable)
        path = os.path.join(
            self.base_path,
            "ForceSMIP_Tier1_final",
            "ForceSMIP-estimates-Tier1"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        if variable != "zmta":
            data_est = np.zeros((30, len(test_models), 876, 72, 144), dtype=np.float32)
        else:
            data_est = np.zeros((22, len(test_models), 876, 17, 72), dtype=np.float32)

        for fname in os.listdir(path):
            if not fname.startswith(f"{var_dir}_"):
                continue
            with nc.Dataset(os.path.join(path, fname)) as ds:
                forced = np.asarray(ds.variables["forced_component"]).squeeze()
                for midx, model in enumerate(test_models):
                    if model in fname:
                        data_est[:, midx] = forced

        # Monthly centering for selected methods
        for idx in methods_to_center:
            if idx >= data_est.shape[0]:
                continue
            for m in range(12):
                mask = (np.arange(data_est.shape[2]) % 12) == m
                mmean = np.nanmean(data_est[idx, :, mask], axis=0)
                data_est[idx, :, mask] -= mmean
        return data_est