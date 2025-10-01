"""
ForceSMIP package initialization.

Exports core API symbols. Some legacy names are kept as shims for backward
compatibility (e.g., compute_yearly_average_dict) and may be removed later.
"""

from .data_loader import ForceSMIPDataLoader
from .preprocessing import (
    yearly_average,
    merge_training_data,
    reshape_training_data,
    capture_nans,
    apply_notnan_filter_complete,
    moving_average_smoothing,
    compute_yearly_average_dict
)
from .algorithms import (
    ridge_regression,
    LowRankSolver,
    WeightedRidgeRegression,   # added for tests
)
from .evaluation import (
    ForceSMIPEvaluator,
    compute_trends_from_data,
    compute_statistics,
)
from .visualization import ForceSMIPVisualizer   # added for tests
from .forcesmip_pipeline import ForceSMIPPipeline, PipelineResult


__all__ = [
    # Core data & preprocessing
    "ForceSMIPDataLoader",
    "yearly_average",
    "merge_training_data",
    "reshape_training_data",
    "capture_nans",
    "apply_notnan_filter_complete",
    "moving_average_smoothing",
    # Algorithms
    "ridge_regression",
    "LowRankSolver",
    "WeightedRidgeRegression",
    # Evaluation / metrics
    "ForceSMIPEvaluator",
    "compute_trends_from_data",
    "compute_statistics",
    # Visualization
    "ForceSMIPVisualizer",
    # Pipeline
    "ForceSMIPPipeline",
    "PipelineResult",
    # Deprecated shim
    "compute_yearly_average_dict",
]