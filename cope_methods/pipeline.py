"""
High-level pipeline orchestrating loading, preprocessing, modeling, and evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch

from .data_loader import ForceSMIPDataLoader
from .preprocessing import (
    compute_yearly_average_dict,
    yearly_average,
    merge_training_data,
    reshape_training_data,
    capture_nans,
    apply_notnan_filter_complete,
    moving_average_smoothing,
)
from .algorithms import ridge_regression, LowRankSolver
from .metrics import compute_trends_from_data
from .evaluation import ForceSMIPEvaluator


@dataclass
class PipelineResult:
    trends_ground_truth: np.ndarray
    trends_methods: Dict[str, np.ndarray]
    comparison: Dict[str, dict]
    nan_indices: List[int]
    kept_indices: List[int]


class ForceSMIPPipeline:
    """
    End-to-end analysis helper.
    """

    def __init__(self, base_path: str, variable: str = "tas"):
        self.base_path = base_path
        self.variable = variable
        self.loader = ForceSMIPDataLoader(base_path)
        self.evaluator = ForceSMIPEvaluator()
        self.longitude = None
        self.latitude = None

    def run(
        self,
        lambda_reg: float = 1000.0,
        rank: int = 10,
        test_models: Optional[List[str]] = None,
        smoothing: bool = False,
        smoothing_window: int = 120,
        trend_slice: slice = slice(30, None),
    ) -> PipelineResult:
        # 1. Load training
        dic_data, dic_forced, lon, lat = self.loader.load_training_data(self.variable)
        self.longitude, self.latitude = lon, lat

        # 2. Load test + ground truth
        data_test, _ = self.loader.load_test_data(self.variable, test_models=test_models)
        ground_truth = self.loader.load_ground_truth(self.variable, test_models=test_models)

        # 3. Yearly aggregation
        dic_data_y = compute_yearly_average_dict(dic_data)
        dic_forced_y = compute_yearly_average_dict(dic_forced)
        x_merge, y_merge = merge_training_data(dic_data_y, dic_forced_y)
        x_dict, y_dict = reshape_training_data(dic_data_y, dic_forced_y)

        # 4. Flatten test monthly
        x_test = torch.from_numpy(data_test.reshape(data_test.shape[0], data_test.shape[1], -1)).float()

        # 5. NaN handling unify
        nan_idx, notnan_idx = capture_nans(x_dict)
        # Extend with test nan columns
        nan_union = set(nan_idx)
        for i in range(x_test.shape[0]):
            mask = torch.isnan(x_test[i]).any(dim=0).cpu().numpy()
            nan_union |= set(np.where(mask)[0])
        notnan_union = sorted(set(range(x_test.shape[2])) - nan_union)

        # 6. Filter
        x_merge_f, y_merge_f, x_dict_f, y_dict_f, x_test_f = apply_notnan_filter_complete(
            x_merge, y_merge, x_dict, y_dict, x_test, notnan_union
        )

        # 7. Optional smoothing
        if smoothing:
            x_test_s = moving_average_smoothing(x_test_f, window_size=smoothing_window)
        else:
            x_test_s = x_test_f

        # 8. Train ridge + low-rank
        W_full = ridge_regression(x_merge_f, y_merge_f, lambda_reg)
        lr_solver = LowRankSolver().fit(x_merge_f, y_merge_f, lambda_reg)
        W_rank = lr_solver.get_rank_k_solution(rank)

        # 9. Predict monthly
        y_pred_full = x_test_f @ W_full
        y_pred_lr = x_test_f @ W_rank
        y_pred_full_s = x_test_s @ W_full
        y_pred_lr_s = x_test_s @ W_rank

        # 10. Ground truth yearly (reshape)
        gt_yearly = yearly_average(ground_truth).reshape(
            ground_truth.shape[0], ground_truth.shape[1] // 12, -1
        )[:, :, notnan_union]

        trends_gt = compute_trends_from_data(gt_yearly, trend_slice)

        def _monthly_to_trends(pred: torch.Tensor) -> np.ndarray:
            arr = pred.cpu().numpy()
            arr_yearly = arr.reshape(arr.shape[0], arr.shape[1] // 12, 12, arr.shape[2]).mean(axis=2)
            return compute_trends_from_data(arr_yearly, trend_slice)

        trends_methods = {
            "Ridge": _monthly_to_trends(y_pred_full),
            "Ridge + Low-rank": _monthly_to_trends(y_pred_lr),
            "Ridge + Smoothing": _monthly_to_trends(y_pred_full_s),
            "Ridge + Low-rank + Smoothing": _monthly_to_trends(y_pred_lr_s),
        }

        comparison = self.evaluator.compare_methods(trends_methods, trends_gt)

        return PipelineResult(
            trends_ground_truth=trends_gt,
            trends_methods=trends_methods,
            comparison=comparison,
            nan_indices=sorted(nan_union),
            kept_indices=notnan_union,
        )