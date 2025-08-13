import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from data_loader import ForceSMIPDataLoader, yearly_average, compute_yearly_average_dict
from preprocessing import merge_training_data, reshape_training_data
from algorithms import WeightedRidgeRegression, ridge_regression, LowRankSolver
from evaluation import ForceSMIPEvaluator, compute_trends_from_data
from visualization import ForceSMIPVisualizer

class ForceSMIPPipeline:
    """Complete pipeline for ForceSMIP challenge analysis with efficient low-rank solutions."""
    
    def __init__(self, base_path: str, variable: str = 'tas'):
        self.base_path = base_path
        self.variable = variable
        self.test_models = ['B', 'D', 'E', 'G', 'J']
        
        # Initialize components
        self.data_loader = ForceSMIPDataLoader(base_path)
        self.weighted_ridge = WeightedRidgeRegression()
        self.evaluator = ForceSMIPEvaluator()
        
        # Low-rank solver for global model
        self.global_lr_solver = LowRankSolver()
        
        # Data storage
        self.data = {}
        self.longitude = None
        self.latitude = None
        
        # Results storage
        self.results = {}
        
    def run_complete_analysis(self, lambda_reg: float = 2500.0, 
                             ranks: List[int] = [5, 10, 15, 20],
                             primary_rank: int = 10,
                             apply_smoothing: bool = False) -> Dict:
        """
        Run the complete analysis pipeline with multiple rank solutions.
        
        Args:
            lambda_reg: Ridge regression regularization parameter
            ranks: List of ranks to compute efficiently
            primary_rank: Primary rank for main analysis
            apply_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Performance summary dictionary
        """
        print("Starting complete ForceSMIP analysis...")
        
        self.load_all_data()
        self.prepare_yearly_data()
        self.train_models(lambda_reg, ranks, primary_rank)
        self.make_predictions(apply_smoothing, ranks, primary_rank)
        self.compute_trends_and_evaluate()
        
        print("Analysis complete!")
        return self.get_performance_summary()
    
    def load_all_data(self) -> None:
        """Load all required datasets."""
        print("Loading training data...")
        (self.data['train_raw'], self.data['train_forced'], 
         self.longitude, self.latitude) = self.data_loader.load_training_data(self.variable)
        
        print("Loading test data...")
        self.data['test_raw'] = self.data_loader.load_test_data(self.variable, self.test_models)
        
        print("Loading ground truth...")
        self.data['test_truth'] = self.data_loader.load_ground_truth(self.variable, self.test_models)
        
        print("Loading other estimates...")
        self.data['estimates'] = self.data_loader.load_estimates(self.variable, self.test_models)
        
        # Initialize visualizer
        self.visualizer = ForceSMIPVisualizer(self.longitude, self.latitude, self.test_models)
        
        print("Data loading complete!")
        
    def prepare_yearly_data(self) -> None:
        """Prepare yearly averaged data."""
        print("Computing yearly averages...")
        
        # Training data
        self.data['train_raw_yearly'] = compute_yearly_average_dict(self.data['train_raw'])
        self.data['train_forced_yearly'] = compute_yearly_average_dict(self.data['train_forced'])
        
        # Test data
        self.data['test_raw_yearly'] = yearly_average(self.data['test_raw'])
        self.data['test_truth_yearly'] = yearly_average(self.data['test_truth'])
        
        # Estimates
        estimates_yearly = np.zeros((self.data['estimates'].shape[0], 
                                   self.data['estimates'].shape[1],
                                   self.data['estimates'].shape[2] // 12,
                                   self.data['estimates'].shape[3],
                                   self.data['estimates'].shape[4]), dtype=np.float32)
        
        for idx_m in range(self.data['estimates'].shape[0]):
            estimates_yearly[idx_m] = yearly_average(self.data['estimates'][idx_m])
        
        self.data['estimates_yearly'] = estimates_yearly
        
        print("Yearly averaging complete!")
        
    def train_models(self, lambda_reg: float = 2500.0, 
                    ranks: List[int] = [5, 10, 15, 20],
                    primary_rank: int = 10) -> None:
        """
        Train ridge regression models with efficient multiple rank solutions.
        
        Args:
            lambda_reg: Regularization parameter
            ranks: List of ranks to compute
            primary_rank: Primary rank for main analysis
        """
        print(f"Training models with lambda={lambda_reg}, ranks={ranks}...")
        
        # Prepare training data
        x_train_yearly, y_train_yearly = merge_training_data(
            self.data['train_raw_yearly'], self.data['train_forced_yearly']
        )
        
        x_yearly, y_yearly = reshape_training_data(
            self.data['train_raw_yearly'], self.data['train_forced_yearly']
        )
        
        # Train global model
        print("Training global ridge regression...")
        w_ridge = ridge_regression(
            x_train_yearly, y_train_yearly, lambda_reg, verbose=True
        )
        
        # Setup efficient low-rank solver
        print("Setting up low-rank solver for global model...")
        self.global_lr_solver.fit(x_train_yearly, y_train_yearly, w_ridge, verbose=True)
        
        # Get multiple rank solutions efficiently
        print(f"Computing solutions for ranks: {ranks}")
        rank_solutions = self.global_lr_solver.get_multiple_ranks(ranks, verbose=True)
        
        # Print explained variance analysis
        print("\nExplained variance analysis:")
        cumulative_var = self.global_lr_solver.cumulative_variance_ratio()
        for rank in ranks:
            if rank <= len(cumulative_var):
                print(f"  Rank {rank}: {cumulative_var[rank-1]:.4f}")
        
        # Store global models
        self.results['w_ridge'] = w_ridge
        self.results['rank_solutions'] = rank_solutions
        self.results['primary_rank'] = primary_rank
        self.results['w_ridge_low_rank'] = rank_solutions[primary_rank]
        
        # Train per-model weighted regression with efficient low-rank
        print("Training per-model weighted regression...")
        performance = self.weighted_ridge.train_per_model(
            x_yearly, y_yearly, lambda_reg, primary_rank, verbose=True
        )
        
        # Get multiple rank solutions for weighted models
        print("Computing multiple rank solutions for weighted models...")
        weighted_rank_solutions = self.weighted_ridge.get_rank_solutions(ranks, verbose=True)
        
        # Compute model weights for different ranks
        weights = {}
        for rank in ranks:
            weights[f'rank_{rank}'] = self.weighted_ridge.compute_weights(
                x_yearly, y_yearly, use_low_rank=True, rank=rank
            )
        
        # Also compute full model weights
        weights['full'] = self.weighted_ridge.compute_weights(
            x_yearly, y_yearly, use_low_rank=False
        )
        
        self.results['model_weights'] = weights
        self.results['training_performance'] = performance
        self.results['weighted_rank_solutions'] = weighted_rank_solutions
        
        print("Model training complete!")
        
    def make_predictions(self, apply_smoothing: bool = False, 
                        ranks: List[int] = [5, 10, 15, 20],
                        primary_rank: int = 10) -> None:
        """
        Make predictions using different methods and ranks.
        
        Args:
            apply_smoothing: Whether to apply temporal smoothing
            ranks: List of ranks to use for predictions
            primary_rank: Primary rank for main analysis
        """
        print("Making predictions...")
        
        # Prepare test data - use MONTHLY data for predictions
        x_test = torch.from_numpy(self.data['test_raw'].reshape(
            self.data['test_raw'].shape[0], self.data['test_raw'].shape[1], -1
        )).to(torch.float32)
        
        print(f"Test data shape: {x_test.shape}")
        
        # Apply smoothing if requested
        if apply_smoothing:
            from preprocessing import gaussian_smoothing
            x_test = gaussian_smoothing(x_test, window_size=25, sigma=5.0)
            print("Applied Gaussian smoothing to test data")
        
        predictions = {}
        
        # Reshape test data for matrix multiplication (flatten first two dimensions)
        x_test_flat = x_test.reshape(-1, x_test.shape[-1])  # (5*876, 10368)
        print(f"Flattened test data shape: {x_test_flat.shape}")
        
        # Global model predictions
        y_pred_flat = x_test_flat @ self.results['w_ridge']
        predictions['ridge'] = y_pred_flat.reshape(x_test.shape[0], x_test.shape[1], -1)
        
        # Multiple rank predictions for global model
        for rank in ranks:
            if rank in self.results['rank_solutions']:
                w_rank = self.results['rank_solutions'][rank]
                y_pred_rank_flat = x_test_flat @ w_rank
                predictions[f'ridge_rank_{rank}'] = y_pred_rank_flat.reshape(x_test.shape[0], x_test.shape[1], -1)
        
        # Weighted predictions for different ranks
        for rank in ranks:
            if f'rank_{rank}' in self.results['model_weights']:
                weights = self.results['model_weights'][f'rank_{rank}']
                # For weighted predictions, we need to use the individual test data format
                y_pred_weighted = self.weighted_ridge.predict_weighted(
                    x_test_flat, weights, use_low_rank=True, rank=rank
                )
                predictions[f'weighted_rank_{rank}'] = y_pred_weighted.reshape(x_test.shape[0], x_test.shape[1], -1)
        
        # Full weighted prediction
        weights_full = self.results['model_weights']['full']
        y_pred_weighted_full = self.weighted_ridge.predict_weighted(
            x_test_flat, weights_full, use_low_rank=False
        )
        predictions['weighted_full'] = y_pred_weighted_full.reshape(x_test.shape[0], x_test.shape[1], -1)
        
        # Store primary predictions for compatibility
        predictions['ridge_yearly'] = predictions[f'ridge_rank_{primary_rank}']
        predictions['weighted_lr_yearly'] = predictions[f'weighted_rank_{primary_rank}']
        
        self.results['predictions'] = predictions
        
        print(f"Generated {len(predictions)} prediction variants")
        print(f"Example prediction shape: {predictions['ridge'].shape}")
        
    def compute_trends_and_evaluate(self, trend_years: slice = slice(30, None)) -> None:
        """Compute trends and evaluate using ForceSMIP metrics."""
        print("Computing trends and evaluating...")
        
        # Compute trends for ground truth
        y_test_yearly = torch.from_numpy(self.data['test_truth_yearly'])
        trend_forced_response = compute_trends_from_data(
            y_test_yearly.numpy(), trend_years
        )
        
        # Compute trends for all predictions
        trends = {}
        for pred_name, pred_data in self.results['predictions'].items():
            # Convert to yearly and compute trends
            pred_numpy = pred_data.cpu().numpy() if isinstance(pred_data, torch.Tensor) else pred_data
            pred_yearly = pred_numpy.reshape(pred_numpy.shape[0], pred_numpy.shape[1] // 12, 12, pred_numpy.shape[2])
            pred_yearly = pred_yearly.mean(axis=2)
            trends[pred_name] = compute_trends_from_data(pred_yearly, trend_years)
        
        # Compute trends for estimates
        trends_estimates = {}
        for idx_m in range(self.data['estimates_yearly'].shape[0]):
            trends_estimates[f'estimate_{idx_m}'] = compute_trends_from_data(
                self.data['estimates_yearly'][idx_m], trend_years
            )
        
        # Evaluate all methods
        all_trends = {**trends, **trends_estimates}
        comparison_results = self.evaluator.compare_methods(all_trends, trend_forced_response)
        
        self.results['trends'] = {
            'ground_truth': trend_forced_response,
            'predictions': trends,
            'estimates': trends_estimates
        }
        self.results['evaluation'] = comparison_results
        
        print("Evaluation complete!")
        
    def analyze_rank_performance(self, ranks: List[int] = None) -> Dict:
        """
        Analyze performance across different ranks.
        
        Args:
            ranks: List of ranks to analyze
            
        Returns:
            Dictionary with rank performance analysis
        """
        if ranks is None:
            ranks = [5, 10, 15, 20]
            
        if 'evaluation' not in self.results:
            print("Please run compute_trends_and_evaluate first!")
            return {}
        
        rank_analysis = {}
        
        # Global model analysis
        global_analysis = {}
        for rank in ranks:
            method_name = f'ridge_rank_{rank}'
            if method_name in self.results['evaluation']:
                global_analysis[rank] = {
                    'mean_nrmse': self.results['evaluation'][method_name]['mean_nrmse'],
                    'mean_pattern_corr': self.results['evaluation'][method_name]['mean_pattern_corr'],
                    'worst_nrmse': self.results['evaluation'][method_name]['worst_nrmse']
                }
        
        # Weighted model analysis
        weighted_analysis = {}
        for rank in ranks:
            method_name = f'weighted_rank_{rank}'
            if method_name in self.results['evaluation']:
                weighted_analysis[rank] = {
                    'mean_nrmse': self.results['evaluation'][method_name]['mean_nrmse'],
                    'mean_pattern_corr': self.results['evaluation'][method_name]['mean_pattern_corr'],
                    'worst_nrmse': self.results['evaluation'][method_name]['worst_nrmse']
                }
        
        rank_analysis['global'] = global_analysis
        rank_analysis['weighted'] = weighted_analysis
        
        # Find optimal ranks
        if global_analysis:
            best_global_rank = min(global_analysis.keys(), 
                                 key=lambda x: global_analysis[x]['mean_nrmse'])
            rank_analysis['best_global_rank'] = best_global_rank
        
        if weighted_analysis:
            best_weighted_rank = min(weighted_analysis.keys(), 
                                   key=lambda x: weighted_analysis[x]['mean_nrmse'])
            rank_analysis['best_weighted_rank'] = best_weighted_rank
        
        return rank_analysis
    
    def plot_results(self, run_idx: int = 0) -> None:
        """Plot comparison results."""
        if 'trends' not in self.results:
            print("Please run compute_trends_and_evaluate first!")
            return
        
        # Plot triple comparison using primary rank
        trend_forced = self.results['trends']['ground_truth']
        trend_pred = self.results['trends']['predictions']['weighted_lr_yearly']
        
        # Compute trend for test member
        x_test_yearly = torch.from_numpy(self.data['test_raw_yearly'])
        trend_test = compute_trends_from_data(x_test_yearly.numpy(), slice(30, None))
        
        fig = self.visualizer.plot_triple_comparison(
            trend_forced, trend_pred, trend_test, run_idx=run_idx
        )
        
        return fig
        
    def get_performance_summary(self) -> Dict:
        """Get summary of performance metrics for primary methods."""
        if 'evaluation' not in self.results:
            print("Please run compute_trends_and_evaluate first!")
            return {}
        
        # Focus on key methods for summary
        key_methods = ['ridge_yearly', 'weighted_lr_yearly', 'weighted_full']
        
        summary = {}
        for method in key_methods:
            if method in self.results['evaluation']:
                results = self.results['evaluation'][method]
                summary[method] = {
                    'mean_nrmse': results['mean_nrmse'],
                    'mean_pattern_corr': results['mean_pattern_corr'],
                    'worst_nrmse': results['worst_nrmse']
                }
        
        return summary