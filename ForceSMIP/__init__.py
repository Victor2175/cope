"""
ForceSMIP Challenge Analysis Package

This package provides a complete pipeline for analyzing the ForceSMIP challenge data,
including data loading, preprocessing, model training, evaluation, and visualization.
"""

from data_loader import ForceSMIPDataLoader, yearly_average, compute_yearly_average_dict
from preprocessing import (merge_training_data, reshape_training_data, stack_models_and_runs,
                           moving_average_smoothing, exponential_smoothing, gaussian_smoothing)
from algorithms import ridge_regression, low_rank_approximation, WeightedRidgeRegression, compute_trend
from evaluation import ForceSMIPEvaluator, compute_statistics, compute_trends_from_data
from visualization import ForceSMIPVisualizer
from forcesmip_pipeline import ForceSMIPPipeline

__all__ = [
    'ForceSMIPDataLoader', 
    'ForceSMIPPipeline',
    'ForceSMIPEvaluator',
    'ForceSMIPVisualizer',
    'WeightedRidgeRegression',
    'yearly_average',
    'compute_yearly_average_dict',
    'merge_training_data',
    'reshape_training_data',
    'stack_models_and_runs',
    'moving_average_smoothing',
    'exponential_smoothing', 
    'gaussian_smoothing',
    'ridge_regression',
    'low_rank_approximation',
    'compute_trend',
    'compute_statistics',
    'compute_trends_from_data'
]