import sys, os
import numpy as np
import torch
import pytest

# Add ForceSMIP module folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ForceSMIP"))

from utils import to_xarray, index_to_latlon  # UPDATED import
from forcesmip_pipeline import ForceSMIPPipeline


class MockLoader:
    def __init__(self, base_path):
        pass

    def load_training_data(self, variable):
        lon = np.linspace(0, 30, 6)
        lat = np.linspace(-25, 25, 5)
        runs, months = 2, 24
        data = {
            "M1": np.random.randn(runs, months, lat.size, lon.size).astype("float32"),
            "M2": np.random.randn(runs, months, lat.size, lon.size).astype("float32"),
        }
        # .reshape(runs, months, lat.size, lon.size)
        forced = {k: v.mean(axis=0).repeat(runs, axis=0).reshape(runs, months, lat.size, lon.size) for k, v in data.items()}
        return data, forced, lon, lat

    def load_test_data(self, variable, tier="Tier1", test_models=None):
        lon = np.linspace(0, 30, 6)
        lat = np.linspace(-25, 25, 5)
        months = 24
        test_models = test_models or ["T1", "T2"]
        test = np.random.randn(len(test_models), months, lat.size, lon.size).astype("float32")
        tm = np.zeros_like(test)
        return test, tm

    def load_ground_truth(self, variable, test_models=None):
        lon = np.linspace(0, 30, 6)
        lat = np.linspace(-25, 25, 5)
        months = 24
        test_models = test_models or ["T1", "T2"]
        return np.random.randn(len(test_models), months, lat.size, lon.size).astype("float32")

    def load_estimates(self, *a, **k):
        return None


def test_index_to_latlon():
    # Arrays
    lat = np.array([-10, 0, 10])
    lon = np.array([0, 90, 180])
    # Flatten rule: idx = lat_idx * n_lon + lon_idx
    # Choose idx=4 -> lat_idx=1, lon_idx=1 (value lat=0, lon=90)
    lat_idx, lon_idx = index_to_latlon(4, lon, lat)
    assert lat_idx == 1
    assert lon_idx == 1
    # Additional check: last element idx= (2 * 3 + 2) = 8
    lat_idx2, lon_idx2 = index_to_latlon(8, lon, lat)
    assert lat_idx2 == 2 and lon_idx2 == 2


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 3,
    reason="Skip on unsupported older GPU"
)
def test_pipeline_run(monkeypatch):
    import forcesmip_pipeline as fp_mod
    monkeypatch.setattr(fp_mod, "ForceSMIPDataLoader", MockLoader)
    pipe = ForceSMIPPipeline(base_path=".", variable="tas")
    result = pipe.run(lambda_reg=10.0, rank=2, smoothing=False, trend_slice=slice(0, None))
    assert result.trends_ground_truth.ndim == 2
    assert "Ridge" in result.trends_methods


def test_to_xarray():
    arr = np.random.randn(2, 24, 5, 6).astype("float32")
    da, lon, lat = to_xarray(arr, ["A", "B"], np.linspace(0, 50, 6), np.linspace(-25, 25, 5))
    # dims: model, time, latitude, longitude
    assert set(da.dims) == {"model", "time", "latitude", "longitude"}
    assert da.shape == (2, 24, 5, 6)