import numpy as np
import torch
from typing import Dict, Tuple, List, Optional, Union

def ridge_regression(x: torch.Tensor, y: torch.Tensor, lambda_reg: float, 
                    surface: Optional[torch.Tensor] = None,
                    verbose: bool = False) -> torch.Tensor:
    """
    Solve ridge regression using closed-form solution.
    
    Args:
        x: Input features (n_samples, n_features) or (n_models, n_times, n_features)
        y: Target values (n_samples, n_targets) or (n_models, n_times, n_targets)
        lambda_reg: Regularization parameter
        surface: Optional surface area weights (latitude, longitude)
        verbose: Whether to print progress
        
    Returns:
        w: Ridge regression weights (n_features, n_targets)
    """
    if verbose:
        print(f"Solving ridge regression with λ={lambda_reg}")
        print(f"Input shape: X={x.shape}, Y={y.shape}")
    
    # Reshape to 2D if needed (flatten first two dimensions)
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])  # (n_models*n_times, n_features)
    if y.dim() == 3:
        y = y.reshape(-1, y.shape[-1])  # (n_models*n_times, n_targets)

    
    if verbose:
        print(f"Reshaped: X={x.shape}, Y={y.shape}")

    # Apply surface weights if provided
    if surface is not None:
        S = torch.diag(surface)
    else:
        S = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)

    # Solve (X^T X + λI)^{-1} X^T Y
    XtX = x.T @ x
    XtY = x.T @ y
    I = torch.eye(XtX.shape[0], dtype=XtX.dtype, device=XtX.device)
    w = torch.linalg.solve(XtX + lambda_reg * S, XtY)
    
    if verbose:
        print(f"Ridge weights computed: {w.shape}")
    
    return w

class LowRankSolver:
    """
    Efficient low-rank approximation solver using precomputed SVD.
    Computes SVD once and provides solutions for any rank.
    """
    
    def __init__(self):
        self.W_full = None
        self.U = None
        self.S = None
        self.Vt = None
        self.XtY = None
        self.is_fitted = False

    def fit(self, x: torch.Tensor, y: torch.Tensor, lambda_reg: float = 1.0,
            surface: Optional[torch.Tensor] = None,
            verbose: bool = False) -> None:
        """
        Compute and store SVD components for efficient rank-k solutions.
        
        Args:
            x: Input features 
            y: Target values
            w_full: Full ridge regression solution
            lambda_reg: Regularization parameter
            verbose: Whether to print progress
        """
        if verbose:
            print("Computing SVD for low-rank approximations...")
            
        # Ensure inputs are 2D
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        if y.dim() == 3:
            y = y.reshape(-1, y.shape[-1])

        # Apply surface weights if provided
        if surface is not None:
            S = torch.diag(surface)
            S_inv = torch.diag(1.0 / surface)
            
        else:
            S = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
            S_inv = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)

        # save w_full
        if self.W_full is None:
            self.W_full = ridge_regression(x @ torch.sqrt(S_inv), y, lambda_reg, surface=surface, verbose=verbose)

        # build the concatenation of (X, sqrt{lambda} I_p)
        I_p = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
        x_augmented = torch.cat([x @ torch.sqrt(S_inv), torch.sqrt(torch.tensor(lambda_reg, dtype=x.dtype, device=x.device)) * I_p], dim=0)

        # Compute SVD of the full solution
        self.U, self.S, self.Vt = torch.linalg.svd(x_augmented @ self.W_full, full_matrices=False)
        self.XtY = x.T @ y
        
        if verbose:
            print(f"SVD computed: U={self.U.shape}, S={self.S.shape}, Vt={self.Vt.shape}")
            print(f"Singular values range: {self.S.min():.6f} to {self.S.max():.6f}")
            
        self.is_fitted = True

    def get_rank_k_solution(self, rank: int, surface: Optional[torch.Tensor] = None, verbose: bool = False) -> torch.Tensor:
        """
        Get rank-k approximation using precomputed SVD.

        Args:
            rank: Desired rank
            surface: Optional surface weights
            verbose: Whether to print progress

        Returns:
            w_k: Rank-k approximated weights
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before getting solutions")

        if rank > min(self.U.shape[1], self.Vt.shape[0]):
            rank = min(self.U.shape[1], self.Vt.shape[0])
            if verbose:
                print(f"Rank reduced to maximum possible: {rank}")

        if verbose:
            print(f"Computing rank-{rank} approximation...")

        # Apply surface weights if provided
        if surface is not None:
            S = torch.diag(surface)
            S_inv = torch.diag(1.0 / surface)
        else:
            S = torch.eye(self.Vt.shape[0], dtype=self.Vt.dtype, device=self.Vt.device)
            S_inv = torch.eye(self.Vt.shape[0], dtype=self.Vt.dtype, device=self.Vt.device)

        # Compute the rank-k weight matrix using SVD components.
        # - S_inv: Inverse of surface weights (or identity)
        # - self.W_full: Full ridge regression weights
        # - self.Vt[:rank, :]: Top-k right singular vectors
        # The multiplication projects the weights onto the top-k singular vectors for low-rank approximation.
        w_k = torch.sqrt(S_inv) @ self.W_full @ self.Vt[:rank, :].T @ self.Vt[:rank, :]  # <-- COMMENTED LINE

        if verbose:
            print(f"Rank-{rank} solution computed: {w_k.shape}")

        return w_k
    
    def get_multiple_ranks(self, ranks: List[int], 
                           surface: Optional[torch.Tensor] = None,
                           verbose: bool = False) -> Dict[int, torch.Tensor]:
        """
        Get solutions for multiple ranks efficiently.
        
        Args:
            ranks: List of desired ranks
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping rank to solution
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before getting solutions")
            
        solutions = {}
        max_rank = min(self.U.shape[1], self.Vt.shape[0])
        
        for rank in ranks:
            if rank > max_rank:
                if verbose:
                    print(f"Warning: Rank {rank} > max possible {max_rank}, skipping")
                continue

            solutions[rank] = self.get_rank_k_solution(rank, surface=surface, verbose=False)

        if verbose:
            print(f"Computed solutions for ranks: {list(solutions.keys())}")
            
        return solutions
    
    def explained_variance_ratio(self) -> torch.Tensor:
        """Get explained variance ratio for each component."""
        if not self.is_fitted:
            raise ValueError("Must call fit() before getting variance ratios")
            
        return self.S**2 / torch.sum(self.S**2)
    
    def cumulative_variance_ratio(self) -> torch.Tensor:
        """Get cumulative explained variance ratio."""
        return torch.cumsum(self.explained_variance_ratio(), dim=0)

def low_rank_approximation(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, 
                          rank: int, verbose: bool = False) -> torch.Tensor:
    """
    Legacy function for backward compatibility.
    Creates a LowRankSolver and returns single rank solution.
    """
    solver = LowRankSolver()
    solver.fit(x, y, w, verbose=verbose)
    return solver.get_rank_k_solution(rank, verbose=verbose)

class WeightedRidgeRegression:
    """Ridge regression with model weighting and efficient low-rank solutions."""
    
    def __init__(self):
        self.models = {}
        self.low_rank_solvers = {}
        self.performance = {}
        
    def train_per_model(self, x_dict: Dict[str, torch.Tensor], 
                       y_dict: Dict[str, torch.Tensor],
                       lambda_reg: float, rank: int = 10,
                       surface: Optional[torch.Tensor] = None,
                       verbose: bool = False) -> Dict[str, Dict]:
        """
        Train ridge regression for each model with low-rank preparation.
        
        Args:
            x_dict: Dictionary of input features per model
            y_dict: Dictionary of targets per model
            lambda_reg: Regularization parameter
            rank: Primary rank for low-rank approximation
            verbose: Whether to print progress
            
        Returns:
            Dictionary of performance metrics per model
        """
        performance = {}
        
        for model_name in x_dict.keys():
            if verbose:
                print(f"\nTraining model: {model_name}")
                
            x_model = x_dict[model_name]
            y_model = y_dict[model_name]

            # Ensure 2D tensors for individual model training
            if x_model.dim() == 3:
                x_model = x_model.reshape(-1, x_model.shape[-1])
            if y_model.dim() == 3:
                y_model = y_model.reshape(-1, y_model.shape[-1])
            
            # Train full ridge regression
            w_full = ridge_regression(x_model, y_model, lambda_reg,surface=surface, verbose=verbose)
            
            # Setup low-rank solver
            lr_solver = LowRankSolver()
            lr_solver.fit(x_model, y_model, lambda_reg=lambda_reg, surface=surface, verbose=verbose)

            # Get primary rank solution
            w_lr = lr_solver.get_rank_k_solution(rank, surface=surface, verbose=verbose)

            # Store models and solvers
            self.models[model_name] = {
                'full': w_full,
                'low_rank': w_lr,
                'rank': rank
            }
            self.low_rank_solvers[model_name] = lr_solver
            
            # Compute training performance
            y_pred_full = x_model @ w_full
            y_pred_lr = x_model @ w_lr
            
            mse_full = torch.mean((y_model - y_pred_full)**2).item()
            mse_lr = torch.mean((y_model - y_pred_lr)**2).item()
            
            performance[model_name] = {
                'mse_full': mse_full,
                'mse_low_rank': mse_lr,
                'rank': rank,
                'explained_variance': lr_solver.cumulative_variance_ratio()[rank-1].item()
            }
            
            if verbose:
                print(f"  Full MSE: {mse_full:.6f}")
                print(f"  Rank-{rank} MSE: {mse_lr:.6f}")
                print(f"  Explained variance: {performance[model_name]['explained_variance']:.4f}")
        
        self.performance = performance
        return performance
    
    def get_rank_solutions(self, ranks: List[int], 
                           surface: Optional[torch.Tensor] = None,
                           verbose: bool = False) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Get solutions for multiple ranks for all models efficiently.
        
        Args:
            ranks: List of ranks to compute
            verbose: Whether to print progress
            
        Returns:
            Nested dictionary: model_name -> rank -> solution
        """
        all_solutions = {}
        
        for model_name, solver in self.low_rank_solvers.items():
            if verbose:
                print(f"Computing multiple ranks for {model_name}...")

            model_solutions = solver.get_multiple_ranks(ranks, surface=surface, verbose=verbose)
            all_solutions[model_name] = model_solutions
            
        return all_solutions
    
    def compute_weights(self, x_dict: Dict[str, torch.Tensor], 
                       y_dict: Dict[str, torch.Tensor],
                       use_low_rank: bool = False, rank: Optional[int] = None,
                       uniform: bool = False) -> torch.Tensor:
        """
        Compute model weights based on performance, with flexible rank selection.
        
        Args:
            x_dict: Dictionary of input features per model
            y_dict: Dictionary of targets per model  
            use_low_rank: Whether to use low-rank solutions
            rank: Specific rank to use (if different from training rank)
            
        Returns:
            Model weights tensor
        """
        model_names = list(self.models.keys())
        weights = (1/len(model_names)) * torch.ones(len(model_names))
        
        if uniform == False:
            for i, model_name in enumerate(model_names):
                x_val = x_dict[model_name]
                y_val = y_dict[model_name]
                
                # Ensure 2D tensors
                if x_val.dim() == 3:
                    x_val = x_val.reshape(-1, x_val.shape[-1])
                if y_val.dim() == 3:
                    y_val = y_val.reshape(-1, y_val.shape[-1])
                
                if use_low_rank:
                    if rank is not None and rank != self.models[model_name]['rank']:
                        # Get solution for specific rank
                        w_model = self.low_rank_solvers[model_name].get_rank_k_solution(rank)
                    else:
                        # Use pre-computed low-rank solution
                        w_model = self.models[model_name]['low_rank']
                else:
                    w_model = self.models[model_name]['full']
                    
                # Compute validation error
                y_pred = x_val @ w_model
                mse = torch.mean((y_val - y_pred)**2)
                
                # Weight inversely proportional to error
                weights[i] = 1.0 / (mse + 1e-8)
            
        # Normalize weights
        weights = weights / torch.sum(weights)
        return weights
    
    def predict_weighted(self, x_test: torch.Tensor, weights: torch.Tensor, 
                        use_low_rank: bool = False, rank: Optional[int] = None) -> torch.Tensor:
        """
        Make weighted predictions with flexible rank selection.
        
        Args:
            x_test: Test input features
            weights: Model weights
            use_low_rank: Whether to use low-rank solutions
            rank: Specific rank to use
            
        Returns:
            Weighted predictions
        """
        model_names = list(self.models.keys())
        
        # Get the shape for output from the first model
        first_model = model_names[0]
        if use_low_rank:
            if rank is not None and rank != self.models[first_model]['rank']:
                w_sample = self.low_rank_solvers[first_model].get_rank_k_solution(rank)
            else:
                w_sample = self.models[first_model]['low_rank']
        else:
            w_sample = self.models[first_model]['full'] 
        
        for i, model_name in enumerate(model_names):
            if use_low_rank:
                if rank is not None and rank != self.models[model_name]['rank']:
                    w_model = self.low_rank_solvers[model_name].get_rank_k_solution(rank)
                else:
                    w_model = self.models[model_name]['low_rank']
            else:
                w_model = self.models[model_name]['full']
                
            y_pred_model = x_test @ w_model

            if i == 0:
                y_pred_weighted = weights[i] * y_pred_model
            else:
                y_pred_weighted += weights[i] * y_pred_model
            
        return y_pred_weighted
    

    def optimize_lambda_cv(self, x_train_dict, y_train_dict, lambda_values=None, 
                          cv_folds=5, objective='worst_case', verbose=True):
        """
        Optimize lambda using cross-validation on the training data.
        
        Args:
            x_train_dict: Dictionary of training inputs per model
            y_train_dict: Dictionary of training targets per model
            lambda_values: List of lambda values to test
            cv_folds: Number of cross-validation folds
            objective: Optimization objective ('worst_case', 'mean', 'variance')
            verbose: Whether to print progress
            
        Returns:
            Dictionary with optimization results
        """
        if verbose:
            print("Optimizing lambda using cross-validation for weighted ridge regression...")
        
        # Merge all training data for CV
        from preprocessing import merge_training_data
        x_merged, y_merged = merge_training_data(x_train_dict, y_train_dict)
        
        # Perform cross-validation
        cv_results = cross_validation_lambda_optimization(
            x_merged, y_merged, 
            lambda_values=lambda_values,
            cv_folds=cv_folds,
            objective=objective,
            verbose=verbose
        )
        
        # Store results
        self.cv_optimization_results = cv_results
        self.optimal_lambda = cv_results['best_lambda']
        
        if verbose:
            print(f"Optimal lambda for weighted ridge regression: {self.optimal_lambda}")
            
        return cv_results
    
    def train_with_optimal_lambda(self, x_train_dict, y_train_dict, rank=None, 
                                 cv_folds=5, objective='worst_case', verbose=True):
        """
        Train weighted ridge regression with cross-validation optimized lambda.
        
        Args:
            x_train_dict: Dictionary of training inputs per model
            y_train_dict: Dictionary of training targets per model
            rank: Low-rank approximation rank (optional)
            cv_folds: Number of cross-validation folds
            objective: Optimization objective for lambda selection
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training performance
        """
        # First optimize lambda
        cv_results = self.optimize_lambda_cv(
            x_train_dict, y_train_dict, 
            cv_folds=cv_folds, 
            objective=objective, 
            verbose=verbose
        )
        
        optimal_lambda = cv_results['best_lambda']
        
        # Train with optimal lambda
        if verbose:
            print(f"\nTraining with optimal lambda = {optimal_lambda}")
        
        performance = self.train_per_model(
            x_train_dict, y_train_dict, 
            lambda_reg=optimal_lambda, 
            rank=rank, 
            verbose=verbose
        )
        
        return {
            'cv_results': cv_results,
            'training_performance': performance,
            'optimal_lambda': optimal_lambda
        }

def compute_trend(data: np.ndarray, years: slice = slice(30, None)) -> np.ndarray:
    """
    Compute linear trends over specified years.
    
    Args:
        data: Time series data (time, ...)
        years: Slice object specifying which years to use
        
    Returns:
        Linear trend coefficients
    """
    data_subset = data[years]
    n_years = data_subset.shape[0]
    
    # Create time vector
    time = np.arange(n_years)
    
    # Reshape for matrix operations
    original_shape = data_subset.shape
    data_flat = data_subset.reshape(n_years, -1)
    
    # Compute trends using least squares
    X = np.column_stack([np.ones(n_years), time])
    coeffs = np.linalg.lstsq(X, data_flat, rcond=None)[0]
    trends = coeffs[1].reshape(original_shape[1:])  # Get slope coefficients
    
    return trends


def cross_validation_lambda_optimization(x_train_dict, y_train_dict, lambda_values=None, surface=None,
                                       cv_folds=5, objective='worst_case',
                                       verbose=True, random_state=42):
    """
    Perform cross-validation to optimize lambda with respect to worst-case objective.
    
    Args:
        x_train: Training input data (n_samples, n_features)
        y_train: Training target data (n_samples, n_targets)
        lambda_values: List of lambda values to test
        surface: Surface area weights (n_features,)
        cv_folds: Number of cross-validation folds
        objective: Optimization objective ('worst_case', 'mean', 'variance')
        verbose: Whether to print progress
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with optimization results
    """
    import numpy as np
    from sklearn.model_selection import KFold
    
    if lambda_values is None:
        lambda_values = [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
    
    np.random.seed(random_state)

    # we do leave one out validation on the training set    
    cv_results = {}
    
    if verbose:
        print(f"Cross-validation with {cv_folds} folds for {len(lambda_values)} lambda values...")
        print(f"Optimization objective: {objective}")
    
    errors_mean = np.zeros(len(lambda_values), dtype=np.float32)
    errors_q75 = np.zeros(len(lambda_values), dtype=np.float32)
    errors_q90 = np.zeros(len(lambda_values), dtype=np.float32)
    errors_q95 = np.zeros(len(lambda_values), dtype=np.float32)
    errors_worst = np.zeros(len(lambda_values), dtype=np.float32) 

    for idx_lambda, lambda_reg in enumerate(lambda_values):

        if verbose:
            print(f"  Testing λ = {lambda_reg}")


        fold_mean_trend_nrmse = np.zeros(len(x_train_dict.keys()), dtype=np.float32)
        fold_q75_trend_nrmse = np.zeros(len(x_train_dict.keys()), dtype=np.float32)
        fold_q90_trend_nrmse = np.zeros(len(x_train_dict.keys()), dtype=np.float32)
        fold_q95_trend_nrmse = np.zeros(len(x_train_dict.keys()), dtype=np.float32)
        fold_worst_trend_nrmse = np.zeros(len(x_train_dict.keys()), dtype=np.float32)

        for idx_m, m in enumerate(x_train_dict.keys()):

            # build the training set as the concatenation of all models except the one being validated
            x_fold_train = np.concatenate([x_train_dict[k] for j, k in enumerate(x_train_dict.keys()) if j != idx_m], axis=0)
            x_fold_train = torch.from_numpy(x_fold_train.reshape(-1, x_fold_train.shape[-1]))
            y_fold_train = np.concatenate([y_train_dict[k] for j, k in enumerate(y_train_dict.keys()) if j != idx_m], axis=0)
            y_fold_train = torch.from_numpy(y_fold_train.reshape(-1, y_fold_train.shape[-1]))

            # build the test set
            x_fold_test = x_train_dict[m]
            y_fold_test = y_train_dict[m]

            # Train model
            w_fold = ridge_regression(x_fold_train, y_fold_train, lambda_reg, surface=surface, verbose=False)

            # Predict on validation set
            y_pred_fold = x_fold_test @ w_fold
            
            # compute the trend for each sample
            trend_pred = np.zeros((y_pred_fold.shape[0], y_pred_fold.shape[2]))
            trends_ground_truth = np.zeros((y_pred_fold.shape[0], y_pred_fold.shape[2]))
            for j in range(y_pred_fold.shape[0]):
                trend_pred[j,:] = np.polyfit(np.arange(y_pred_fold.shape[1]), y_pred_fold[j,:,:], 1)[0]
                trends_ground_truth[j,:] = np.polyfit(np.arange(y_fold_test.shape[1]), y_fold_test[j,:,:], 1)[0]


            # compute the normalized RMSE of the trend

            nmrse = np.sqrt(np.nansum((trend_pred - trends_ground_truth)**2,axis=1))/np.sqrt(np.nansum(trends_ground_truth**2, axis=1))


            fold_mean_trend_nrmse[idx_m] = np.mean(nmrse)
            fold_q75_trend_nrmse[idx_m] = np.nanquantile(nmrse, 0.75)
            fold_q90_trend_nrmse[idx_m] = np.nanquantile(nmrse, 0.90)
            fold_q95_trend_nrmse[idx_m] = np.nanquantile(nmrse, 0.95)
            fold_worst_trend_nrmse[idx_m] = np.nanmax(nmrse)



        # Aggregate results only for the MEAN over runs
        errors_mean[idx_lambda] = np.mean(fold_mean_trend_nrmse) 
        errors_q75[idx_lambda] = np.nanquantile(fold_mean_trend_nrmse, 0.75)
        errors_q90[idx_lambda] = np.nanquantile(fold_mean_trend_nrmse, 0.90)
        errors_q95[idx_lambda] = np.nanquantile(fold_mean_trend_nrmse, 0.95)
        errors_worst[idx_lambda] = np.max(fold_mean_trend_nrmse)  # Shape: (n_folds, n_spatial_locations)



        cv_results[lambda_reg] = {
            'mean_nrmse': errors_mean[idx_lambda],
            'q75_nrmse': errors_q75[idx_lambda],
            'q90_nrmse': errors_q90[idx_lambda],
            'q95_nrmse': errors_q95[idx_lambda],
            'worst_nrmse': errors_worst[idx_lambda],
            'nrmse_variance': np.nanvar(fold_mean_trend_nrmse)
        }
        
        if verbose:
            print(f"    Mean NRMSE: {cv_results[lambda_reg]['mean_nrmse']:.4f}")
            print(f"   Q75 NRMSE: {cv_results[lambda_reg]['q75_nrmse']:.4f}")
            print(f"   Q90 NRMSE: {cv_results[lambda_reg]['q90_nrmse']:.4f}")
            print(f"   Q95 NRMSE: {cv_results[lambda_reg]['q95_nrmse']:.4f}")
            print(f"    Worst NRMSE: {cv_results[lambda_reg]['worst_nrmse']:.4f}")
            print(f"    NRMSE Variance: {cv_results[lambda_reg]['nrmse_variance']:.4f}")
    
    # Select best lambda based on objective
    if objective == 'worst_case':
        best_lambda = min(cv_results.keys(), key=lambda x: cv_results[x]['worst_nrmse'])
        optimization_metric = 'worst_nrmse'
    elif objective == 'mean':
        best_lambda = min(cv_results.keys(), key=lambda x: cv_results[x]['mean_nrmse'])
        optimization_metric = 'mean_nrmse'
    elif objective == 'variance':
        best_lambda = min(cv_results.keys(), key=lambda x: cv_results[x]['nrmse_variance'])
        optimization_metric = 'nrmse_variance'
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    if verbose:
        print(f"\nOptimization complete!")
        print(f"Best λ = {best_lambda} (optimizing {optimization_metric})")
        print(f"Best {optimization_metric}: {cv_results[best_lambda][optimization_metric]:.4f}")
    
    return {
        'best_lambda': best_lambda,
        'optimization_metric': optimization_metric,
        'cv_results': cv_results,
        'lambda_values': lambda_values,
        'cv_folds': cv_folds,
        'objective': objective
    }



def cross_validation_lambda_rank_optimization(x_train_dict, y_train_dict, lambda_values=None, rank_values=None, surface=None,
                                       cv_folds=5, objective='worst_case', 
                                       verbose=True, random_state=42):
    """
    Perform cross-validation to optimize (lambda, rank) with respect to worst-case objective.

    Args:
        x_train: Training input data (n_samples, n_features)
        y_train: Training target data (n_samples, n_targets)
        lambda_values: List of lambda values to test
        rank_values: List of rank values to test
        cv_folds: Number of cross-validation folds
        objective: Optimization objective ('worst_case', 'mean', 'variance')
        verbose: Whether to print progress
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with optimization results
    """
    import numpy as np
    from sklearn.model_selection import KFold
    
    if lambda_values is None:
        lambda_values = [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]

    if rank_values is None:
        rank_values = [5, 10, 15, 20, 25, 30, 50, 100]

    np.random.seed(random_state)

    # we do leave one out validation on the training set    
    cv_results = {}
    
    if verbose:
        print(f"Cross-validation with {cv_folds} folds for {len(lambda_values)} lambda values...")
        print(f"Optimization objective: {objective}")

    errors_mean = np.zeros((len(lambda_values), len(rank_values)), dtype=np.float32)
    errors_q75 = np.zeros((len(lambda_values), len(rank_values)), dtype=np.float32)
    errors_q90 = np.zeros((len(lambda_values), len(rank_values)), dtype=np.float32)
    errors_q95 = np.zeros((len(lambda_values), len(rank_values)), dtype=np.float32)
    errors_worst = np.zeros((len(lambda_values), len(rank_values)), dtype=np.float32)

    for idx_lambda, lambda_reg in enumerate(lambda_values):

        if verbose:
            print(f"  Testing λ = {lambda_reg}")


        fold_mean_trend_nrmse = np.zeros((len(rank_values), len(x_train_dict.keys())), dtype=np.float32)
        fold_q75_trend_nrmse = np.zeros((len(rank_values), len(x_train_dict.keys())), dtype=np.float32)
        fold_q90_trend_nrmse = np.zeros((len(rank_values), len(x_train_dict.keys())), dtype=np.float32)
        fold_q95_trend_nrmse = np.zeros((len(rank_values), len(x_train_dict.keys())), dtype=np.float32)
        fold_worst_trend_nrmse = np.zeros((len(rank_values), len(x_train_dict.keys())), dtype=np.float32)

        for idx_m, m in enumerate(x_train_dict.keys()):

            # build the training set as the concatenation of all models except the one being validated
            x_fold_train = np.concatenate([x_train_dict[k] for j, k in enumerate(x_train_dict.keys()) if j != idx_m], axis=0)
            x_fold_train = torch.from_numpy(x_fold_train.reshape(-1, x_fold_train.shape[-1]))
            y_fold_train = np.concatenate([y_train_dict[k] for j, k in enumerate(y_train_dict.keys()) if j != idx_m], axis=0)
            y_fold_train = torch.from_numpy(y_fold_train.reshape(-1, y_fold_train.shape[-1]))

            # build the test set
            x_fold_test = x_train_dict[m]
            y_fold_test = y_train_dict[m]

            # Train model
            # w_fold = ridge_regression(x_fold_train, y_fold_train, lambda_reg, verbose=False)

            # Setup low-rank solver for efficient multiple rank solutions
            print(f"\nSetting up low-rank solver...")
            lr_solver = LowRankSolver()
            lr_solver.fit(x_fold_train, y_fold_train, lambda_reg=lambda_reg, surface=surface, verbose=True)

            # Get solutions for multiple ranks efficiently
            rank_solutions = lr_solver.get_multiple_ranks(rank_values, verbose=True)

            nrmse = np.zeros((len(rank_solutions), x_fold_test.shape[0]), dtype=np.float32)

            # Predict on validation set
            for idx_r, (r, w) in enumerate(rank_solutions.items()):
                y_pred_fold = x_fold_test @ w

                # compute the trend for each sample
                trend_pred = np.zeros((y_pred_fold.shape[0], y_pred_fold.shape[2]))
                trends_ground_truth = np.zeros((y_pred_fold.shape[0], y_pred_fold.shape[2]))
                for j in range(y_pred_fold.shape[0]):
                    trend_pred[j,:] = np.polyfit(np.arange(y_pred_fold.shape[1]), y_pred_fold[j,:,:], 1)[0]
                    trends_ground_truth[j,:] = np.polyfit(np.arange(y_fold_test.shape[1]), y_fold_test[j,:,:], 1)[0]


                # compute the normalized RMSE of the trend
                nrmse[idx_r, :] = np.sqrt(np.nansum((trend_pred - trends_ground_truth)**2,axis=1))/np.sqrt(np.nansum(trends_ground_truth**2, axis=1))

            fold_mean_trend_nrmse[:,idx_m] = np.mean(nrmse,axis=1)
            fold_q75_trend_nrmse[:,idx_m] = np.nanquantile(nrmse, 0.75, axis=1)
            fold_q90_trend_nrmse[:,idx_m] = np.nanquantile(nrmse, 0.90, axis=1)
            fold_q95_trend_nrmse[:,idx_m] = np.nanquantile(nrmse, 0.95, axis=1)
            fold_worst_trend_nrmse[:,idx_m] = np.nanmax(nrmse, axis=1)



        # Aggregate results only for the MEAN over runs
        errors_mean[idx_lambda,:] = np.mean(fold_mean_trend_nrmse, axis=1)
        errors_q75[idx_lambda,:] = np.nanquantile(fold_mean_trend_nrmse, 0.75, axis=1)
        errors_q90[idx_lambda,:] = np.nanquantile(fold_mean_trend_nrmse, 0.90, axis=1)
        errors_q95[idx_lambda,:] = np.nanquantile(fold_mean_trend_nrmse, 0.95, axis=1)
        errors_worst[idx_lambda,:] = np.max(fold_mean_trend_nrmse, axis=1)  # Shape: (n_folds, n_spatial_locations)

        

        for idx_r, r in enumerate(rank_values):
            cv_results[lambda_reg, r] = {
                'mean_nrmse': errors_mean[idx_lambda, idx_r],
                'q75_nrmse': errors_q75[idx_lambda, idx_r],
                'q90_nrmse': errors_q90[idx_lambda, idx_r],
                'q95_nrmse': errors_q95[idx_lambda, idx_r],
                'worst_nrmse': errors_worst[idx_lambda, idx_r]
            }
        
            if verbose:
                print(f"    Mean NRMSE: {cv_results[lambda_reg, r]['mean_nrmse']:.4f}")
                print(f"   Q75 NRMSE: {cv_results[lambda_reg, r]['q75_nrmse']:.4f}")
                print(f"   Q90 NRMSE: {cv_results[lambda_reg, r]['q90_nrmse']:.4f}")
                print(f"   Q95 NRMSE: {cv_results[lambda_reg, r]['q95_nrmse']:.4f}")
                print(f"    Worst NRMSE: {cv_results[lambda_reg, r]['worst_nrmse']:.4f}")
                # print(f"    NRMSE Variance: {cv_results[lambda_reg, r]['nrmse_variance']:.4f}")
    
    # Select best lambda based on objective
    if objective == 'worst_case':
        (best_lambda,best_rank) = min(cv_results.keys(), key=lambda x: cv_results[x]['worst_nrmse'])
        optimization_metric = 'worst_nrmse'
    elif objective == 'mean':
        (best_lambda,best_rank) = min(cv_results.keys(), key=lambda x: cv_results[x]['mean_nrmse'])
        optimization_metric = 'mean_nrmse'
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    if verbose:
        print(f"\nOptimization complete!")
        print(f"Best λ = {best_lambda} (optimizing {optimization_metric})")
        print(f"Best {optimization_metric}: {cv_results[best_lambda, best_rank][optimization_metric]:.4f}")

    return {
        'best_lambda': best_lambda,
        'best_rank': best_rank,
        'optimization_metric': optimization_metric,
        'cv_results': cv_results,
        'lambda_values': lambda_values,
        'rank_values': rank_values,
        'cv_folds': cv_folds,
        'objective': objective
    }

def plot_cv_lambda_results(cv_optimization_results, figsize=(15, 5)):
    """
    Plot cross-validation results for lambda optimization.
    
    Args:
        cv_optimization_results: Results from cross_validation_lambda_optimization
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    cv_results = cv_optimization_results['cv_results']
    best_lambda = cv_optimization_results['best_lambda']
    objective = cv_optimization_results['objective']
    
    lambdas = list(cv_results.keys())
    mean_nrmse = [cv_results[l]['mean_nrmse'] for l in lambdas]
    worst_nrmse = [cv_results[l]['worst_nrmse'] for l in lambdas]
    nrmse_variance = [cv_results[l]['nrmse_variance'] for l in lambdas]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Mean NRMSE
    ax1.semilogx(lambdas, mean_nrmse, 'o-', color='blue', label='Mean NRMSE')
    ax1.axvline(best_lambda, color='red', linestyle='--', alpha=0.7, label=f'Best λ = {best_lambda}')
    ax1.set_xlabel('Regularization Parameter (λ)')
    ax1.set_ylabel('Mean NRMSE')
    ax1.set_title('Cross-Validation: Mean NRMSE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Worst NRMSE
    ax2.semilogx(lambdas, worst_nrmse, 'o-', color='red', label='Worst NRMSE')
    ax2.axvline(best_lambda, color='red', linestyle='--', alpha=0.7, label=f'Best λ = {best_lambda}')
    ax2.set_xlabel('Regularization Parameter (λ)')
    ax2.set_ylabel('Worst Case NRMSE')
    ax2.set_title('Cross-Validation: Worst Case NRMSE')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Highlight if optimizing worst case
    if objective == 'worst_case':
        ax2.set_facecolor('#fff0f0')
        ax2.set_title('Cross-Validation: Worst Case NRMSE ⭐', fontweight='bold')
    
    # Plot 3: NRMSE Variance
    ax3.semilogx(lambdas, nrmse_variance, 'o-', color='green', label='NRMSE Variance')
    ax3.axvline(best_lambda, color='red', linestyle='--', alpha=0.7, label=f'Best λ = {best_lambda}')
    ax3.set_xlabel('Regularization Parameter (λ)')
    ax3.set_ylabel('NRMSE Variance')
    ax3.set_title('Cross-Validation: NRMSE Variance')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    return fig


def plot_cv_lambda_rank_results(cv_optimization_results, figsize=(15, 5)):
    """
    Plot cross-validation results for lambda optimization.

    Args:
        cv_optimization_results: Results from cross_validation_lambda_optimization
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    cv_results = cv_optimization_results['cv_results']
    best_lambda = cv_optimization_results['best_lambda']
    objective = cv_optimization_results['objective']

    lambdas_rank_values = list(cv_results.keys())
    lambdas = sorted(list(set([lambda_reg for (lambda_reg, _) in lambdas_rank_values])))
    rank_values = sorted(list(set([rank for (_, rank) in lambdas_rank_values])))

    mean_nrmse = np.zeros((len(lambdas), len(rank_values)), dtype=np.float32)
    worst_nrmse = np.zeros((len(lambdas), len(rank_values)), dtype=np.float32)

    for (i, l) in enumerate(lambdas):
        for (j, r) in enumerate(rank_values):
            mean_nrmse[i, j] = cv_results[(l, r)]['mean_nrmse']
            worst_nrmse[i, j] = cv_results[(l, r)]['worst_nrmse']

    X, Y = np.meshgrid(lambdas, rank_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Use perceptually uniform colormaps
    cmap = "plasma"

    c1 = ax1.pcolormesh(X, Y, mean_nrmse.T, shading='auto', cmap=cmap, edgecolors='k', linewidth=0.2)
    fig.colorbar(c1, ax=ax1, label='Mean NRMSE')
    ax1.set_title('Cross-Validation: Mean NRMSE')
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Rank (r)')
    ax1.set_yscale('log')
    ax1.set_xticks(lambdas)
    ax1.set_xticklabels([str(l) for l in lambdas])
    ax1.set_yticks(rank_values)
    ax1.set_yticklabels([str(r) for r in rank_values])
    ax1.grid(True, alpha=0.3)

    c2 = ax2.pcolormesh(X, Y, worst_nrmse.T, shading='auto', cmap=cmap, edgecolors='k', linewidth=0.2)
    fig.colorbar(c2, ax=ax2, label='Worst NRMSE')
    ax2.set_title('Cross-Validation: Worst NRMSE')
    ax2.set_xlabel('Lambda (λ)')
    ax2.set_ylabel('Rank (r)')
    ax2.set_yscale('log')
    ax2.set_xticks(lambdas)
    ax2.set_xticklabels([str(l) for l in lambdas])
    ax2.set_yticks(rank_values)
    ax2.set_yticklabels([str(r) for r in rank_values])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig