import sys
import os
# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../ForceSMIP")))
import pytest
import importlib
import ForceSMIP

def test_all_symbols_present():
    for symbol in ForceSMIP.__all__:
        assert hasattr(ForceSMIP, symbol), f"{symbol} not found in module"

def test_import_classes_and_functions():
    ForceSMIP.ForceSMIPDataLoader
    ForceSMIP.ForceSMIPPipeline
    ForceSMIP.ForceSMIPEvaluator
    ForceSMIP.ForceSMIPVisualizer
    ForceSMIP.WeightedRidgeRegression
    ForceSMIP.yearly_average
    ForceSMIP.ridge_regression

def test_force_smip_data_loader_instantiation():
    try:
        loader = ForceSMIP.ForceSMIPDataLoader()
    except TypeError:
        pass

def test_force_smip_pipeline_instantiation():
    try:
        pipeline = ForceSMIP.ForceSMIPPipeline()
    except TypeError:
        pass

def test_yearly_average_callable():
    with pytest.raises(TypeError):
        ForceSMIP.yearly_average()

def test_ridge_regression_callable():
    with pytest.raises(TypeError):
        ForceSMIP.ridge_regression()