#!/usr/bin/env python3
"""
Example script demonstrating the ForceSMIP pipeline with efficient low-rank analysis.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from forcesmip_pipeline import ForceSMIPPipeline

def main():
    # Configuration
    base_path = '/net/krypton/climdyn_nobackup/FTP'
    variable = 'tas'  # Surface air temperature
    
    # Analysis parameters
    lambda_reg = 2500.0
    ranks_to_test = [5, 10, 15, 20, 25, 30]
    primary_rank = 10
    
    print(f"Initializing ForceSMIP pipeline for {variable}")
    print(f"Testing ranks: {ranks_to_test}")
    print(f"Primary rank: {primary_rank}")
    
    # Initialize pipeline
    pipeline = ForceSMIPPipeline(base_path, variable=variable)
    
    # Run complete analysis with multiple ranks
    try:
        summary = pipeline.run_complete_analysis(
            lambda_reg=lambda_reg,
            ranks=ranks_to_test,
            primary_rank=primary_rank,
            apply_smoothing=False
        )
        
        print("\n" + "="*60)
        print("                ANALYSIS COMPLETE!")
        print("="*60)
        
        # Display primary performance summary
        print("\n=== Primary Methods Performance ===")
        for method, metrics in summary.items():
            print(f"\n{method}:")
            print(f"  Mean NRMSE: {metrics['mean_nrmse']:.4f}")
            print(f"  Mean Pattern Correlation: {metrics['mean_pattern_corr']:.4f}")
            print(f"  Worst NRMSE: {metrics['worst_nrmse']:.4f}")
        
        # Analyze rank performance
        print("\n=== Rank Performance Analysis ===")
        rank_analysis = pipeline.analyze_rank_performance(ranks_to_test)
        
        if 'global' in rank_analysis:
            print("\nGlobal Ridge Regression by Rank:")
            for rank, metrics in rank_analysis['global'].items():
                print(f"  Rank {rank}: NRMSE={metrics['mean_nrmse']:.4f}, "
                      f"Corr={metrics['mean_pattern_corr']:.4f}")
        
        if 'weighted' in rank_analysis:
            print("\nWeighted Ridge Regression by Rank:")
            for rank, metrics in rank_analysis['weighted'].items():
                print(f"  Rank {rank}: NRMSE={metrics['mean_nrmse']:.4f}, "
                      f"Corr={metrics['mean_pattern_corr']:.4f}")
        
        # Report optimal ranks
        if 'best_global_rank' in rank_analysis:
            print(f"\nOptimal global rank: {rank_analysis['best_global_rank']}")
        if 'best_weighted_rank' in rank_analysis:
            print(f"Optimal weighted rank: {rank_analysis['best_weighted_rank']}")
        
        # Explained variance information
        if hasattr(pipeline.global_lr_solver, 'cumulative_variance_ratio'):
            print("\n=== Explained Variance by Rank ===")
            cumulative_var = pipeline.global_lr_solver.cumulative_variance_ratio()
            for rank in ranks_to_test:
                if rank <= len(cumulative_var):
                    print(f"  Rank {rank}: {cumulative_var[rank-1]:.4f}")
        
        print("\n" + "="*60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease check that the data paths are correct.")
        print("Expected data structure:")
        print("  /net/krypton/climdyn_nobackup/FTP/ForceSMIP/Training-Ext/Amon/tas/")
        print("  /net/krypton/climdyn_nobackup/FTP/ForceSMIP_Tier1_final/Evaluation-Tier1/")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())