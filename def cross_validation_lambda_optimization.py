def cross_validation_lambda_optimization(x_train, y_train, lambda_values=None, 
                                       cv_folds=5, objective='worst_case', 
                                       verbose=True, random_state=42):
    """
    Perform cross-validation to optimize lambda with respect to worst-case objective.
    
    Args:
        x_train: Training input data (n_samples, n_features)
        y_train: Training target data (n_samples, n_targets)
        lambda_values: List of lambda values to test
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
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_results = {}
    
    if verbose:
        print(f"Cross-validation with {cv_folds} folds for {len(lambda_values)} lambda values...")
        print(f"Optimization objective: {objective}")
    
    for lambda_reg in lambda_values:
        fold_errors = []
        
        if verbose:
            print(f"  Testing λ = {lambda_reg}")
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x_train)):
            # Split data
            x_fold_train = x_train[train_idx]
            y_fold_train = y_train[train_idx]
            x_fold_val = x_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Train model
            w_fold = ridge_regression(x_fold_train, y_fold_train, lambda_reg, verbose=False)
            
            # Predict on validation set
            y_pred_fold = x_fold_val @ w_fold
            
            # Compute normalized RMSE for each spatial location
            fold_nrmse = []
            for i in range(y_fold_val.shape[1]):  # For each spatial location
                y_true_i = y_fold_val[:, i]
                y_pred_i = y_pred_fold[:, i]
                
                # Compute NRMSE
                rmse = np.sqrt(np.mean((y_true_i - y_pred_i)**2))
                y_range = np.max(y_true_i) - np.min(y_true_i)
                nrmse = rmse / (y_range + 1e-8)  # Add small epsilon to avoid division by zero
                fold_nrmse.append(nrmse)
            
            fold_errors.append(fold_nrmse)
        
        # Aggregate results across folds
        fold_errors = np.array(fold_errors)  # Shape: (n_folds, n_spatial_locations)
        
        # Compute statistics for this lambda
        mean_nrmse_per_location = np.mean(fold_errors, axis=0)
        
        cv_results[lambda_reg] = {
            'mean_nrmse': np.mean(mean_nrmse_per_location),
            'worst_nrmse': np.max(mean_nrmse_per_location),
            'nrmse_variance': np.var(mean_nrmse_per_location),
            'fold_errors': fold_errors,
            'mean_nrmse_per_location': mean_nrmse_per_location
        }
        
        if verbose:
            print(f"    Mean NRMSE: {cv_results[lambda_reg]['mean_nrmse']:.4f}")
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