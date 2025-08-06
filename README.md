# ForceSMIP Challenge Analysis

This repository contains a comprehensive analysis pipeline for the ForceSMIP (Forced Response Model Intercomparison Project) challenge, focusing on extracting forced climate response signals from noisy climate model data.

## Repository Structure

```
/home/vcohen/cope/
├── src/                           # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # Data loading and preprocessing utilities
│   ├── preprocessing.py          # Data preprocessing and smoothing functions
│   ├── algorithms.py             # Ridge regression and machine learning algorithms
│   ├── evaluation.py             # Evaluation metrics and performance assessment
│   ├── visualization.py          # Plotting and visualization functions
│   └── forcesmip_pipeline.py     # Main pipeline orchestrating the analysis
├── run_analysis.py               # Example script to run complete analysis
├── forcesmip6_tas.ipynb          # Original notebook with exploratory analysis
└── (other existing files)        # Additional notebooks and scripts
```

## Module Descriptions

### `src/data_loader.py`
- **ForceSMIPDataLoader**: Main class for loading ForceSMIP datasets
- Functions for loading training data, test data, ground truth, and estimates
- Handles monthly centering and data preprocessing
- Supports yearly averaging computations

### `src/preprocessing.py`
- Data merging and reshaping utilities
- Smoothing functions (moving average, exponential, Gaussian)
- Training data preparation for machine learning models
- Model stacking and run concatenation functions

### `src/algorithms.py`
- **Ridge regression** implementation with closed-form solution
- **Low-rank approximation** methods
- **WeightedRidgeRegression** class for multi-model ensemble predictions
- Training loss computation and model weight optimization

### `src/evaluation.py`
- **ForceSMIPEvaluator** class implementing challenge metrics:
  - Normalized RMSE computation
  - Amplitude ratio calculation
  - Pattern correlation assessment
- Performance comparison across multiple methods
- Statistical analysis utilities

### `src/visualization.py`
- **ForceSMIPVisualizer** class for creating publication-quality plots
- Robinson projection maps for global climate data
- Multi-panel comparison visualizations
- Time series plotting functions
- Performance comparison charts

### `src/forcesmip_pipeline.py`
- **ForceSMIPPipeline**: Complete analysis orchestration class
- Methods for:
  - Data loading and preparation
  - Model training and optimization
  - Prediction generation
  - Performance evaluation
  - Results visualization

## Quick Start

### Basic Usage

```python
from src.forcesmip_pipeline import ForceSMIPPipeline

# Initialize pipeline
base_path = '/net/krypton/climdyn_nobackup/FTP/ForceSMIP'
pipeline = ForceSMIPPipeline(base_path, variable='tas')

# Run complete analysis
summary = pipeline.run_complete_analysis(
    lambda_reg=2500.0,    # Ridge regression regularization
    rank=10,              # Low-rank approximation rank
    apply_smoothing=False # Optional data smoothing
)

# View results
print(summary)
```

### Running from Command Line

```bash
# Run the example analysis script
python run_analysis.py
```

### Custom Analysis

```python
from src import *

# Load data manually
data_loader = ForceSMIPDataLoader(base_path)
train_raw, train_forced, lon, lat = data_loader.load_training_data('tas')

# Train custom models
evaluator = ForceSMIPEvaluator()
visualizer = ForceSMIPVisualizer(lon, lat, ['B', 'D', 'E', 'G', 'J'])

# Use individual components as needed
```

## Key Features

### Data Handling
- **Multi-format support**: NetCDF4 climate data files
- **Automatic preprocessing**: Monthly centering, yearly averaging
- **Memory efficient**: Chunked loading for large datasets
- **Multiple variables**: Temperature (tas), ocean temperature (tos)

### Machine Learning Methods
- **Ridge Regression**: L2-regularized linear regression
- **Low-rank Methods**: SVD-based dimensionality reduction
- **Ensemble Learning**: Weighted combination of multiple models
- **Smoothing Techniques**: Multiple temporal smoothing options

### Evaluation Metrics
- **Pattern Correlation**: Uncentered spatial correlation
- **Normalized RMSE**: Trend amplitude-normalized error
- **Amplitude Ratio**: Predicted vs. true signal strength
- **Statistical Summaries**: Comprehensive performance statistics

### Visualization
- **Global Maps**: Robinson projection climate visualizations
- **Comparison Plots**: Side-by-side method comparisons
- **Time Series**: Temporal evolution analysis
- **Performance Charts**: Method ranking and statistics

## Requirements

```python
# Core dependencies
numpy >= 1.20.0
torch >= 1.9.0
matplotlib >= 3.3.0
cartopy >= 0.18.0
netCDF4 >= 1.5.0

# Optional for enhanced functionality
scikit-learn >= 0.24.0
```

## Data Requirements

The pipeline expects ForceSMIP data organized as:
```
/path/to/data/
├── Training-Ext/Amon/tas/          # Training data by model
├── ForceSMIP_Tier1_final/
│   ├── Evaluation-Tier1/           # Test data
│   ├── ensmeans-Tier1/             # Ground truth ensemble means
│   └── ForceSMIP-estimates-Tier1/  # Other method estimates
```

## Example Analysis Results

The pipeline produces:
- **Performance Rankings**: Comparison of methods across metrics
- **Global Trend Maps**: Spatial patterns of climate trends
- **Statistical Summaries**: Comprehensive evaluation statistics
- **Visualization Suite**: Publication-ready figures

## Contributing

When adding new functionality:
1. Follow the modular structure
2. Add comprehensive docstrings
3. Include type hints
4. Update this README
5. Add examples in docstrings

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:
[Add appropriate citation information]
