# ForceSMIP Challenge Analysis

End-to-end pipeline for extracting forced climate response signals (ForceSMIP). Includes ridge regression, low‑rank approximation, smoothing, cross‑validation, benchmarking, and visualization.

## Updated Repository Structure

```
.
├── ForceSMIP/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── algorithms.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── forcesmip_pipeline.py
│   ├── utils.py (grid helpers if added)
│   └── grid_utils.py (2.5° grid, if separated)
├── forcesmip6_tas_refactored.ipynb      # Main refactored analysis
├── learn_the_whole_dataset.ipynb        # Full SST dataset experiment
├── forcesmip6_tas.ipynb                 # Legacy exploratory notebook
├── run_analysis.py
├── tests/                               # Pytest unit tests
├── results/                             # Generated CSV metrics (e.g. trend_rmse_results.csv)
├── figures/                             # Saved plots
├── requirements.txt
├── .github/workflows/ci.yml             # GitHub Actions CI (tests + notebook)
└── README.md
```

## Key Modules

### ForceSMIP/data_loader.py
- ForceSMIPDataLoader: loads training, test, ground truth, and benchmark estimate fields.
- Handles monthly to yearly aggregation hooks (used downstream).

### ForceSMIP/preprocessing.py
- merge_training_data / reshape_training_data
- NaN handling: capture_nans, filtering helpers
- Smoothing: moving_average_smoothing (and others if defined)
- Utility filters: filter_training_dict_by_indices, filter_training_data_by_indices

### ForceSMIP/algorithms.py
- ridge_regression (closed form, optional surface weighting)
- LowRankSolver (efficient multi-rank weight truncation + variance explained)
- WeightedRidgeRegression (per‑model weighting)
- Cross‑validation:
  - cross_validation_lambda_optimization
  - cross_validation_lambda_rank_optimization
  - plot_cv_lambda_results / plot_cv_lambda_rank_results

### ForceSMIP/evaluation.py
- ForceSMIPEvaluator.compare_methods: mean/worst NRMSE, pattern correlation, amplitude ratio
- compute_trends_from_data
- compute_statistics

### ForceSMIP/visualization.py
- Robinson projection maps (with cartopy)
- Triple comparison plots
- Performance bar / ranking charts

### Grid Utilities (utils.py or grid_utils.py)
- generate_grid_with_areas(res_deg=2.5, lon_domain="0-360")
- plot_empty_grid() for diagnostic grid visualization
- Returns centers, edges, and spherical cell areas (used for surface weighting)

### forcesmip_pipeline.py
High-level orchestration:
- load → preprocess → train → low-rank → predict → evaluate → visualize

## Recent Additions / Changes
- Migration from `src/` to `ForceSMIP/` package path (update imports).
- Added 2.5° × 2.5° regular grid generation + cell area weighting option.
- Added cross‑validation (lambda and (lambda, rank)).
- Refactored notebook `forcesmip6_tas_refactored.ipynb` using pipeline components.
- CI workflow (`.github/workflows/ci.yml`) runs pytest and executes notebook; uploads artifacts.
- Added synthetic + full-dataset workflows (`learn_the_whole_dataset.ipynb`).
- Ranking export: `results/trend_rmse_results.csv`.

## Installation

```bash
pip install -r requirements.txt
```

Ensure system libs for cartopy (CI installs: libproj, proj-data, proj-bin, libgeos-dev).

## Quick Start

```python
from ForceSMIP.forcesmip_pipeline import ForceSMIPPipeline

base_path = "/path/to/ForceSMIP/data"
pipeline = ForceSMIPPipeline(base_path, variable="tos")
summary = pipeline.run_complete_analysis(
    lambda_reg=5000.0,
    rank=10,
    apply_smoothing=False
)
print(summary)
```

### Manual Component Usage

```python
from ForceSMIP.data_loader import ForceSMIPDataLoader
from ForceSMIP.preprocessing import merge_training_data, capture_nans
from ForceSMIP.algorithms import ridge_regression, LowRankSolver
from ForceSMIP.evaluation import ForceSMIPEvaluator, compute_trends_from_data

loader = ForceSMIPDataLoader(base_path)
dic_data, dic_forced, lon, lat = loader.load_training_data("tos")

X,Y = merge_training_data(dic_data, dic_forced)
w = ridge_regression(X, Y, lambda_=5000.0)

solver = LowRankSolver()
solver.W_full = w
solver.fit(X, Y, 5000.0)
rank_W = solver.get_multiple_ranks([10])[10]
```

### 2.5° Grid Example

```python
from ForceSMIP.grid_utils import generate_grid_with_areas
lat_c, lon_c, lat_e, lon_e, cell_areas = generate_grid_with_areas(res_deg=2.5)
```

## Cross-Validation Example

```python
from ForceSMIP.algorithms import cross_validation_lambda_rank_optimization
cv = cross_validation_lambda_rank_optimization(
    x_train_dict, y_train_dict,
    lambda_values=[1000, 5000, 10000],
    rank_values=[5, 10, 20],
    cv_folds=5,
    objective="worst_case",
    verbose=True
)
print(cv["best_lambda"], cv["best_rank"])
```

## CI

GitHub Actions:
- Path: `.github/workflows/ci.yml`
- Runs pytest + executes main notebook
- Uploads executed notebook + results CSV

## Data Layout (Expected)

```
/data_root/
├── Training-Ext/Amon/<var>/<model>/*.nc
├── ForceSMIP_Tier1_final/
│   ├── Evaluation-Tier1/<var>/*.nc
│   ├── ensmeans-Tier1/<var>/*.nc
│   └── ForceSMIP-estimates-Tier1/<method>/<var>/*.nc
```

## Outputs
- Figures: `figures/*.png`
- Trends & rankings: `results/trend_rmse_results.csv`
- Low-rank variance diagnostics (printed + optional plots)

## Contributing
1. Add tests in `tests/`
2. Follow type hints & docstrings
3. Update README for structural changes
4. Keep notebook cells reproducible (fixed seeds)

## License
Specify a license (e.g., MIT) in a LICENSE file.

## Citation
(Add citation information)
