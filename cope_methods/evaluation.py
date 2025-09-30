"""
Evaluation wrapper that aggregates metrics per method.
"""

from __future__ import annotations
from typing import Dict
import numpy as np

from .metrics import pattern_correlation, amplitude_ratio


class ForceSMIPEvaluator:
    """
    Compare predicted trend maps against ground truth.
    """

    def compare_methods(
        self,
        method_trends: Dict[str, np.ndarray],
        ground_truth_trends: np.ndarray,
    ) -> Dict[str, dict]:
        """
        Args:
            method_trends: {method: (runs, features)}
            ground_truth_trends: (runs, features)
        """
        results = {}
        for name, pred in method_trends.items():
            nrmse_list, pc_list, ar_list = [], [], []
            for i in range(ground_truth_trends.shape[0]):
                gt = ground_truth_trends[i]
                pr = pred[i]
                nrmse = float(
                    (np.sqrt(np.nansum((pr - gt) ** 2) / np.nansum(gt**2 + 1e-12)))
                )
                pc = pattern_correlation(pr, gt)
                ar = amplitude_ratio(pr, gt)
                nrmse_list.append(nrmse)
                pc_list.append(pc)
                ar_list.append(ar)
            nrmse_arr = np.asarray(nrmse_list)
            pc_arr = np.asarray(pc_list)
            ar_arr = np.asarray(ar_list)
            results[name] = {
                "mean_nrmse": float(nrmse_arr.mean()),
                "mean_pattern_corr": float(pc_arr.mean()),
                "worst_nrmse": float(nrmse_arr.max()),
                "variance_nrmse": float(nrmse_arr.var()),
                "metrics": {
                    "normalized_rmse": nrmse_arr,
                    "pattern_correlation": pc_arr,
                    "amplitude_ratio": ar_arr,
                },
            }
        return results