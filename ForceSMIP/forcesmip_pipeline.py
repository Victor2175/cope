from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import warnings

from data_loader import ForceSMIPDataLoader
from preprocessing import (
    yearly_average,
    merge_training_data,
    reshape_training_data,
    capture_nans,
    apply_notnan_filter_complete,
    moving_average_smoothing,
)
from algorithms import ridge_regression, LowRankSolver
from evaluation import (
    ForceSMIPEvaluator,
    compute_trends_from_data,
)


@dataclass
class PipelineResult:
    trends_ground_truth: np.ndarray
    trends_methods: Dict[str, np.ndarray]          # dict: method -> trend array
    comparison: Dict[str, Any]
    nan_indices: List[int]
    kept_indices: List[int]
    original_months: int
    used_months: int
    weights_full: Optional[np.ndarray]
    weights_rank: Optional[np.ndarray]

    # Backward compatibility aliases
    @property
    def ground_truth_trends(self):
        return self.trends_ground_truth

    @property
    def method_trends(self):
        return self.trends_methods


class ForceSMIPPipeline:
    """
    End‑to‑end ForceSMIP pipeline (ensures time dimension is a multiple of 12).
    """

    def __init__(self, base_path: str, variable: str = "tas"):
        self.base_path = base_path
        self.variable = variable
        self.loader = ForceSMIPDataLoader(base_path)
        self.evaluator = ForceSMIPEvaluator()
        self.longitude: Optional[np.ndarray] = None
        self.latitude: Optional[np.ndarray] = None

    @staticmethod
    def _coerce_1d_to_time_series(arr: np.ndarray) -> np.ndarray:
        """
        Heuristically reshape a 1D flattened array into (runs=1, months, spatial).

        Assumes the 1D vector is a flattening of (months, spatial) with months a multiple of 12.
        Strategy:
          - Enumerate all month candidates (multiples of 12) up to a cap (max_months=2000).
          - Prefer the candidate that yields a spatial dimension between 1 and 100000 and
            whose months value is closest to (len(arr) ** 0.5) * 12 scaling (mild bias toward
            reasonable square-ish decompositions).
          - Fallback: use largest months candidate.

        If no multiple-of-12 candidate, raise ValueError.
        """
        L = arr.size
        if L < 12:
            raise ValueError(f"Cannot coerce 1D array of length {L} into monthly data.")
        candidates = [m for m in range(12, min(L, 2000) + 1, 12) if L % m == 0]
        if not candidates:
            raise ValueError(
                f"1D array length {L} not factorizable into (months * spatial) with months multiple of 12."
            )
        # Heuristic scoring
        best = None
        best_score = None
        target = (L ** 0.5) * 12  # loose heuristic anchor
        for m in candidates:
            spatial = L // m
            if spatial <= 0:
                continue
            score = abs(m - target)  # smaller is better
            if best is None or score < best_score:
                best = (m, spatial)
                best_score = score
        months, spatial = best
        reshaped = arr.reshape(1, months, spatial)
        warnings.warn(
            f"Coerced 1D array of length {L} into shape (1, {months}, {spatial}). "
            "Verify this heuristic reshape is correct."
        )
        return reshaped

    @staticmethod
    def _pad_to_full_years(arr: np.ndarray, target_mult: int = 12) -> np.ndarray:
        """
        Ensure the time axis (assumed axis=1) is a multiple of target_mult (12 months).

        Accepted shapes:
          (runs, months)
          (runs, months, spatial)
          (runs, months, lat, lon)
          1D (flattened) -> heuristically reshaped to (1, months, spatial)

        Strategy:
          - Heuristically expand 1D arrays.
          - If months < target_mult: pad by repeating last month.
          - If months > target_mult but not multiple: truncate to floor multiple.
        """
        if arr.ndim == 1:
            arr = ForceSMIPPipeline._coerce_1d_to_time_series(arr)
        if arr.ndim < 2:
            raise ValueError(f"_pad_to_full_years: cannot find time axis in shape {arr.shape}")

        months = arr.shape[1]
        if months == 0:
            raise ValueError("Empty time dimension.")

        if months < target_mult:
            pad_needed = target_mult - months
            last_slice = arr[:, -1:, ...]
            pad_block = np.repeat(last_slice, pad_needed, axis=1)
            return np.concatenate([arr, pad_block], axis=1)

        full_year_months = (months // target_mult) * target_mult
        if full_year_months != months:
            arr = arr[:, :full_year_months]
        return arr

    @staticmethod
    def _trim_or_pad_dict(d: Dict[str, np.ndarray], allow_short: bool) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in d.items():
            if allow_short:
                out[k] = ForceSMIPPipeline._pad_to_full_years(v)
            else:
                if v.ndim == 1:
                    v = ForceSMIPPipeline._coerce_1d_to_time_series(v)
                months = v.shape[1]
                fy = (months // 12) * 12
                if fy == 0:
                    raise ValueError(
                        f"Time dimension has fewer than 12 months for key '{k}'; cannot compute yearly averages."
                    )
                if fy != months:
                    v = v[:, :fy]
                out[k] = v
        return out

    @staticmethod
    def _trim_or_pad_array(arr: np.ndarray, allow_short: bool) -> np.ndarray:
        if allow_short:
            return ForceSMIPPipeline._pad_to_full_years(arr)
        months = arr.shape[1]
        fy = (months // 12) * 12
        if fy == 0:
            raise ValueError("Time dimension has fewer than 12 months; cannot compute yearly averages.")
        if fy != months:
            arr = arr[:, :fy]
        return arr

    @staticmethod
    def _trim_to_full_years(arr: np.ndarray) -> np.ndarray:
        """
        Trim an array with a monthly time axis (axis=1) so that length is a multiple of 12.
        Expects shape (..., months, ...); here we use (runs/models, months, lat, lon).
        """
        months = arr.shape[1]
        full_year_months = (months // 12) * 12
        if full_year_months == 0:
            raise ValueError("Time dimension has fewer than 12 months; cannot compute yearly averages.")
        if full_year_months != months:
            arr = arr[:, :full_year_months]
        return arr

    @staticmethod
    def _trim_dict_months(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: ForceSMIPPipeline._trim_to_full_years(v) for k, v in d.items()}

    def _monthly_to_yearly(self, arr: np.ndarray) -> np.ndarray:
        # arr: (members, months, features)
        n_years = arr.shape[1] // 12
        trimmed = arr[:, : n_years * 12]
        reshaped = trimmed.reshape(arr.shape[0], n_years, 12, arr.shape[2])
        return np.nanmean(reshaped, axis=2)

    def _pred_to_trends(
        self,
        pred: torch.Tensor,
        keep_idx: List[int],
        trend_slice: slice,
    ) -> np.ndarray:
        pred_np = pred.detach().cpu().numpy()
        yearly = self._monthly_to_yearly(pred_np)
        return compute_trends_from_data(yearly, year_slice=trend_slice)

    def run(
        self,
        lambda_reg: float = 1e3,
        rank: int = 10,
        test_models: Optional[List[str]] = None,
        smoothing: bool = False,
        smoothing_window: int = 120,
        trend_slice: slice = slice(30, None),
        center_add_back: bool = True,
        verbose: bool = False,
        allow_short_series: bool = True,
    ) -> PipelineResult:
        # 1. Load training data
        if verbose:
            print("[Pipeline] Loading training data...")
        dic_data, dic_forced, lon, lat = self.loader.load_training_data(self.variable)
        self.longitude, self.latitude = lon, lat

        # 2. Load test & ground truth
        if verbose:
            print("[Pipeline] Loading test data...")
        data_test, temporal_means = self.loader.load_test_data(
            self.variable, test_models=test_models
        )
        if verbose:
            print("[Pipeline] Loading ground truth...")
        ground_truth = self.loader.load_ground_truth(
            self.variable, test_models=test_models
        )

        original_months = data_test.shape[1]

        # 3. Ensure / pad to multiple of 12
        if verbose:
            action = "Padding/Trimming" if allow_short_series else "Trimming"
            print(f"[Pipeline] {action} to full years (multiple of 12 months)...")

        print(dic_data[list(dic_data.keys())[0]].shape)
        dic_data = self._trim_or_pad_dict(dic_data, allow_short_series)
        print(dic_forced[list(dic_forced.keys())[0]].shape)
        dic_forced = self._trim_or_pad_dict(dic_forced, allow_short_series)
        data_test = self._trim_or_pad_array(data_test, allow_short_series)
        ground_truth = self._trim_or_pad_array(ground_truth, allow_short_series)
        if temporal_means is not None:
            temporal_means = self._trim_or_pad_array(temporal_means, allow_short_series)

        used_months = data_test.shape[1]
        if verbose and used_months != original_months:
            print(f"[Pipeline] Adjusted months from {original_months} → {used_months}")

        # 4. Yearly averages (dict comprehension; compute_yearly_average_dict removed)
        if verbose:
            print("[Pipeline] Yearly averaging training data...")
        dic_data_yearly = {k: yearly_average(v) for k, v in dic_data.items()}
        dic_forced_yearly = {k: yearly_average(v) for k, v in dic_forced.items()}

        print(dic_data_yearly.keys())
        print(dic_forced_yearly[list(dic_forced_yearly.keys())[0]].shape)

        # 5. Merge / reshape
        x_merge, y_merge = merge_training_data(dic_data_yearly, dic_forced_yearly)
        x_dict, y_dict = reshape_training_data(dic_data_yearly, dic_forced_yearly)

        # 6. Prepare flattened test monthly
        x_test = torch.from_numpy(
            data_test.reshape(data_test.shape[0], data_test.shape[1], -1)
        ).float()

        # 7. NaN handling
        nan_idx_train, _ = capture_nans(x_dict)
        nan_union = set(nan_idx_train)
        for m in range(x_test.shape[0]):
            mask_any = torch.isnan(x_test[m]).any(dim=0).cpu().numpy()
            nan_union |= set(np.where(mask_any)[0])
        kept = sorted(set(range(x_test.shape[2])) - nan_union)

        # 8. Apply not-NaN filter
        (
            x_merge_f,
            y_merge_f,
            x_dict_f,
            y_dict_f,
            x_test_f,
        ) = apply_notnan_filter_complete(
            x_merge, y_merge, x_dict, y_dict, x_test, kept
        )

        # 9. Optional smoothing
        if smoothing:
            if verbose:
                print(f"[Pipeline] Smoothing test (window={smoothing_window})...")
            x_test_s = moving_average_smoothing(
                x_test_f, window_size=smoothing_window, mode="same"
            )
        else:
            x_test_s = x_test_f

        # 10. Ridge fit
        if verbose:
            print(f"[Pipeline] Fitting ridge (λ={lambda_reg})...")
        W_full = ridge_regression(x_merge_f, y_merge_f, lambda_reg=lambda_reg)

        # 11. Low-rank
        if verbose:
            print(f"[Pipeline] Computing low-rank (rank={rank})...")
        lr_solver = LowRankSolver()
        lr_solver.fit(
            x_merge_f, y_merge_f, lambda_reg=lambda_reg, surface=None, verbose=False
        )
        W_rank = lr_solver.get_rank_k_solution(rank)

        # 12. Predictions
        y_pred_full = x_test_f @ W_full
        y_pred_rank = x_test_f @ W_rank
        y_pred_full_s = x_test_s @ W_full
        y_pred_rank_s = x_test_s @ W_rank

        # 13. Add back climatology if provided (already trimmed)
        if center_add_back and temporal_means is not None:
            tm_flat = temporal_means.reshape(
                temporal_means.shape[0], temporal_means.shape[1], -1
            )[:, :, kept]
            tm_t = torch.from_numpy(tm_flat).float()
            y_pred_full = y_pred_full + tm_t
            y_pred_rank = y_pred_rank + tm_t
            y_pred_full_s = y_pred_full_s + tm_t
            y_pred_rank_s = y_pred_rank_s + tm_t

        # 14. Ground truth yearly + trends
        gt_yearly = yearly_average(ground_truth)  # (models, years, lat, lon)
        gt_yearly_flat = gt_yearly.reshape(
            gt_yearly.shape[0], gt_yearly.shape[1], -1
        )[:, :, kept]
        trends_gt = compute_trends_from_data(gt_yearly_flat, year_slice=trend_slice)

        # 15. Method trends
        trends_methods = {
            "Ridge": self._pred_to_trends(y_pred_full, kept, trend_slice),
            "Ridge + LowRank": self._pred_to_trends(y_pred_rank, kept, trend_slice),
            "Ridge + Smoothing": self._pred_to_trends(y_pred_full_s, kept, trend_slice),
            "Ridge + LowRank + Smoothing": self._pred_to_trends(
                y_pred_rank_s, kept, trend_slice
            ),
        }

        # 16. Evaluation
        comparison = self.evaluator.compare_methods(trends_methods, trends_gt)

        if verbose:
            print("[Pipeline] Done.")

        return PipelineResult(
            trends_ground_truth=trends_gt,
            trends_methods=trends_methods,   # renamed to match test expectation
            comparison=comparison,
            nan_indices=sorted(nan_union),
            kept_indices=kept,
            original_months=original_months,
            used_months=used_months,
            weights_full=W_full,
            weights_rank=W_rank,
        )