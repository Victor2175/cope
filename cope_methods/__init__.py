"""
cope_methods: Modular ForceSMIP utilities.

Public API re-exports core classes and functions for convenience.
"""

from .constants import VARIABLE_MAP, DEFAULT_TEST_MODELS
from .data_loader import ForceSMIPDataLoader
from .preprocessing import (
    yearly_average,
    compute_yearly_average_dict,
    merge_training_data,
    reshape_training_data,
    capture_nans,
    apply_notnan_filter_complete,
    moving_average_smoothing
)
from .algorithms import (
    ridge_regression,
    LowRankSolver,
    WeightedRidgeRegression
)
from .cross_validation import (
    cross_validation_lambda_optimization,
    cross_validation_lambda_rank_optimization
)
from .evaluation import ForceSMIPEvaluator
from .metrics import (
    compute_trends_from_data,
    pattern_correlation,
    amplitude_ratio,
    compute_statistics
)
from .visualization import ForceSMIPVisualizer
from .pipeline import ForceSMIPPipeline, PipelineResult
from .utils import index_to_latlon_coords

__all__ = [
    "VARIABLE_MAP", "DEFAULT_TEST_MODELS",
    "ForceSMIPDataLoader",
    "yearly_average", "compute_yearly_average_dict",
    "merge_training_data", "reshape_training_data",
    "capture_nans", "apply_notnan_filter_complete",
    "moving_average_smoothing",
    "ridge_regression", "LowRankSolver", "WeightedRidgeRegression",
    "cross_validation_lambda_optimization",
    "cross_validation_lambda_rank_optimization",
    "ForceSMIPEvaluator",
    "compute_trends_from_data", "pattern_correlation",
    "amplitude_ratio", "compute_statistics",
    "ForceSMIPVisualizer",
    "ForceSMIPPipeline", "PipelineResult",
    "index_to_latlon_coords"
]