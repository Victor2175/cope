import os
import sys
import numpy as np
import pytest
import matplotlib

# Use non-interactive backend for CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add ForceSMIP module folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ForceSMIP"))

from evaluation import (
    ForceSMIPEvaluator,
    compute_statistics,
    compute_trends_from_data,
)


@pytest.fixture
def evaluator():
    return ForceSMIPEvaluator()


def test_compute_normalized_rmse_basic(evaluator):
    runs, feats = 5, 10
    true = np.random.randn(runs, feats).astype("float32")
    noise = 0.1 * np.random.randn(runs, feats).astype("float32")
    pred = true + noise
    nrmse = evaluator.compute_normalized_rmse(pred, true)
    assert nrmse.shape == (runs,)
    assert np.all(nrmse >= 0)
    assert nrmse.mean() < 0.5  # Should be small with low noise


def test_amplitude_ratio_positive(evaluator):
    runs, feats = 4, 7
    true = np.random.randn(runs, feats).astype("float32")
    pred = 1.2 * true
    amp = evaluator.compute_amplitude_ratio(pred, true)
    assert amp.shape == (runs,)
    # Amplitude ratio should be near 1.2
    assert np.allclose(amp.mean(), 1.2, atol=0.15)


def test_pattern_correlation_consistency(evaluator):
    runs, feats = 3, 12
    true = np.random.randn(runs, feats).astype("float32")
    pred = true + 0.05 * np.random.randn(runs, feats).astype("float32")
    nrmse = evaluator.compute_normalized_rmse(pred, true)
    gamma = evaluator.compute_amplitude_ratio(pred, true)
    corr = evaluator.compute_pattern_correlation(pred, true)
    # Formula relationship already used internally; just sanity bounds
    assert corr.shape == (runs,)
    assert np.all(corr <= 1.05) and np.all(corr >= -1.05)


def test_evaluate_predictions(evaluator):
    runs, feats = 6, 15
    true = np.random.randn(runs, feats).astype("float32")
    pred = true * 0.9 + 0.1 * np.random.randn(runs, feats).astype("float32")
    metrics = evaluator.evaluate_predictions(pred, true)
    assert set(metrics.keys()) == {"normalized_rmse", "amplitude_ratio", "pattern_correlation"}
    for k, v in metrics.items():
        assert isinstance(v, np.ndarray)
        assert v.shape == (runs,)


def test_compare_methods_structure(evaluator):
    runs, feats = 5, 20
    true = np.random.randn(runs, feats).astype("float32")
    pred_a = true + 0.05 * np.random.randn(runs, feats).astype("float32")
    pred_b = true * 1.1
    comp = evaluator.compare_methods({"A": pred_a, "B": pred_b}, true)
    assert set(comp.keys()) == {"A", "B"}
    required = {"mean_nrmse", "mean_pattern_corr", "worst_nrmse", "variance_nrmse", "metrics"}
    for m in comp.values():
        assert required.issubset(m.keys())
        assert set(m["metrics"].keys()) == {"normalized_rmse", "amplitude_ratio", "pattern_correlation"}


def test_plot_performance_comparison(evaluator):
    # Minimal synthetic comparison dict
    comp = {
        "Method1": {
            "mean_nrmse": 0.3,
            "mean_pattern_corr": 0.95,
            "worst_nrmse": 0.5,
            "variance_nrmse": 0.01,
            "metrics": {
                "normalized_rmse": np.array([0.3, 0.4]),
                "amplitude_ratio": np.array([1.0, 1.1]),
                "pattern_correlation": np.array([0.95, 0.93]),
            },
        },
        "Method2": {
            "mean_nrmse": 0.35,
            "mean_pattern_corr": 0.9,
            "worst_nrmse": 0.55,
            "variance_nrmse": 0.015,
            "metrics": {
                "normalized_rmse": np.array([0.35, 0.45]),
                "amplitude_ratio": np.array([0.9, 1.05]),
                "pattern_correlation": np.array([0.9, 0.88]),
            },
        },
    }
    fig = evaluator.plot_performance_comparison(comp, metric="mean_nrmse")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_compute_statistics():
    data = np.array([1.0, 2.0, 3.0, np.nan, 4.0])
    stats = compute_statistics(data)
    expected_keys = {"mean", "median", "std", "min", "max", "q25", "q75", "q90", "q95", "variance"}
    assert expected_keys.issubset(stats.keys())
    assert np.isclose(stats["mean"], np.nanmean(data))


def test_compute_trends_from_data_basic():
    runs, years, feats = 4, 10, 6
    # Linear trend + noise
    t = np.arange(years)
    base = 0.2 * t
    data = np.stack([
        base[:, None] + 0.01 * np.random.randn(years, feats) for _ in range(runs)
    ]).astype("float32")  # (runs, years, feats)
    trends = compute_trends_from_data(data, year_slice=slice(0, None))
    assert trends.shape == (runs, feats)
    # Expect slope near 0.2
    assert np.allclose(trends.mean(), 0.2, atol=0.05)


def test_compute_trends_from_data_nan_handling():
    runs, years, feats = 3, 12, 8
    data = np.random.randn(runs, years, feats).astype("float32")
    # Insert NaNs in first year for some features (these should propagate to output)
    nan_features = [1, 4, 6]
    data[:, 0, nan_features] = np.nan
    trends = compute_trends_from_data(data, year_slice=slice(0, None))
    assert np.all(np.isnan(trends[:, nan_features]))
    assert np.all(~np.isnan(trends[:, [f for f in range(feats) if f not in nan_features]]))


def test_pattern_correlation_monotonic(evaluator):
    # Perfect correlation vs noisy version
    runs, feats = 2, 50
    true = np.random.randn(runs, feats).astype("float32")
    perfect = true.copy()
    noisy = true + 0.5 * np.random.randn(runs, feats).astype("float32")
    corr_perfect = evaluator.compute_pattern_correlation(perfect, true)
    corr_noisy = evaluator.compute_pattern_correlation(noisy, true)
    # Perfect should have higher (closer to 1)
    assert np.all(corr_perfect >= corr_noisy - 1e-6)