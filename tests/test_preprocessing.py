import sys, os, numpy as np, torch, pytest
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ForceSMIP"))

from data_loader import ForceSMIPDataLoader  # if needed
from preprocessing import (
    yearly_average,
    merge_training_data,
    reshape_training_data,
    capture_nans,
    apply_notnan_filter_complete,
    moving_average_smoothing,
    filter_training_data_by_indices,
)

def test_yearly_average_basic():
    runs, months, lat, lon = 3, 24, 4, 5
    data = np.random.randn(runs, months, lat, lon).astype("float32")
    y = yearly_average(data)
    assert y.shape == (runs, months // 12, lat, lon)

def test_compute_yearly_average_dict_replacement():
    d = {
        "M1": np.random.randn(2, 24, 3, 4).astype("float32"),
        "M2": np.random.randn(2, 24, 3, 4).astype("float32"),
    }
    yd = {k: yearly_average(v) for k, v in d.items()}
    assert set(yd.keys()) == set(d.keys())
    for k, arr in yd.items():
        assert arr.shape[1] == 2

def test_filter_training_data_handles_transposed_targets():
    n_samples = 5
    n_features = 12
    X = torch.randn(n_samples, n_features)
    Y = torch.randn(n_samples, n_features)  # transposed on purpose
    valid = list(range(0, n_features, 2))
    Xf, Yf = filter_training_data_by_indices(X, Y, valid)
    assert Xf.shape == (n_samples, len(valid))
    assert Yf.shape == (n_samples, len(valid))