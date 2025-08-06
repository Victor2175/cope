import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt

class ForceSMIPEvaluator:
    """Class for evaluating ForceSMIP predictions using challenge metrics."""
    
    def __init__(self):
        pass
        
    def compute_normalized_rmse(self, pred_trends: np.ndarray, 
                               true_trends: np.ndarray) -> np.ndarray:
        """
        Compute normalized RMSE for trend patterns.
        
        Args:
            pred_trends: Predicted trend coefficients
            true_trends: True trend coefficients
            
        Returns:
            Normalized RMSE values
        """
        numerator = np.linalg.norm(pred_trends - true_trends, axis=1)
        denominator = np.linalg.norm(true_trends, axis=1)
        
        return numerator / denominator
    
    def compute_amplitude_ratio(self, pred_trends: np.ndarray,
                               true_trends: np.ndarray) -> np.ndarray:
        """
        Compute amplitude ratio between predicted and true trends.
        
        Args:
            pred_trends: Predicted trend coefficients
            true_trends: True trend coefficients
            
        Returns:
            Amplitude ratio values
        """
        pred_amplitude = np.linalg.norm(pred_trends, axis=1)
        true_amplitude = np.linalg.norm(true_trends, axis=1)
        
        return pred_amplitude / true_amplitude
    
    def compute_pattern_correlation(self, pred_trends: np.ndarray,
                                   true_trends: np.ndarray) -> np.ndarray:
        """
        Compute uncentered pattern correlation.
        
        Args:
            pred_trends: Predicted trend coefficients
            true_trends: True trend coefficients
            
        Returns:
            Pattern correlation values
        """
        normalized_rmse = self.compute_normalized_rmse(pred_trends, true_trends)
        amplitude_ratio = self.compute_amplitude_ratio(pred_trends, true_trends)
        
        # Using the relationship: r_i = (1 + γ_i² - nRMSE_i²) / (2γ_i)
        pattern_corr = (1 + amplitude_ratio**2 - normalized_rmse**2) / (2 * amplitude_ratio)
        
        return pattern_corr
    
    def evaluate_predictions(self, pred_trends: np.ndarray,
                           true_trends: np.ndarray) -> Dict:
        """
        Compute all evaluation metrics.
        
        Args:
            pred_trends: Predicted trend coefficients
            true_trends: True trend coefficients
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'normalized_rmse': self.compute_normalized_rmse(pred_trends, true_trends),
            'amplitude_ratio': self.compute_amplitude_ratio(pred_trends, true_trends),
            'pattern_correlation': self.compute_pattern_correlation(pred_trends, true_trends)
        }
        
        return metrics
    
    def compare_methods(self, predictions_dict: Dict[str, np.ndarray],
                       true_trends: np.ndarray) -> Dict:
        """
        Compare multiple prediction methods.
        
        Args:
            predictions_dict: Dictionary mapping method names to predictions
            true_trends: True trend coefficients
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for method_name, pred_trends in predictions_dict.items():
            metrics = self.evaluate_predictions(pred_trends, true_trends)
            
            comparison[method_name] = {
                'mean_nrmse': np.sqrt(np.mean(metrics['normalized_rmse']**2)),
                'mean_pattern_corr': np.sqrt(np.mean(metrics['pattern_correlation']**2)),
                'worst_nrmse': np.max(metrics['normalized_rmse']),
                'variance_nrmse': np.var(metrics['normalized_rmse']),
                'metrics': metrics
            }
            
        return comparison
    
    def plot_performance_comparison(self, comparison_results: Dict,
                                   metric: str = 'mean_nrmse',
                                   title: Optional[str] = None) -> plt.Figure:
        """
        Plot performance comparison across methods.
        
        Args:
            comparison_results: Results from compare_methods
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        methods = list(comparison_results.keys())
        values = [comparison_results[method][metric] for method in methods]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(methods)), values)
        
        ax.set_xlabel('Methods')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title or f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

def compute_statistics(data: np.ndarray) -> Dict:
    """
    Compute comprehensive statistics for performance data.
    
    Args:
        data: Performance data array
        
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.quantile(data, 0.25),
        'q75': np.quantile(data, 0.75),
        'q90': np.quantile(data, 0.90),
        'q95': np.quantile(data, 0.95),
        'variance': np.var(data)
    }

def compute_trends_from_data(data: np.ndarray, year_slice: slice = slice(30, None)) -> np.ndarray:
    """
    Compute linear trends from yearly data.
    
    Args:
        data: Input data of shape (n_runs, n_years, n_features)
        year_slice: Slice of years to use for trend computation
        
    Returns:
        Trend coefficients of shape (n_runs, n_features)
    """
    data_subset = data[:, year_slice, :]
    trends = np.zeros((data_subset.shape[0], data_subset.shape[2]), dtype=np.float32)
    
    for i in range(data_subset.shape[0]):
        trends[i, :] = np.polyfit(np.arange(data_subset.shape[1]), data_subset[i, :, :], 1)[0]

    return trends