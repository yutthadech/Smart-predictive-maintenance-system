"""
NASA C-MAPSS Predictive Maintenance System
==========================================

A production-ready ML pipeline for predicting Remaining Useful Life (RUL)
of turbofan engines using the NASA C-MAPSS dataset.

Modules:
    - data_preprocessing: Load and validate raw data
    - feature_engineer: Compute RUL and engineer features
    - model_training: Train Random Forest models
    - model_evaluation: Evaluate model performance
    - visualizer: Generate plots and visualizations
    - utils: Logging, file handling, and utilities
"""

__version__ = "2.0.0"
__author__ = "AI Engineer"

from .data_preprocessing import load_data, validate_data
from .feature_engineer import engineer_features
from .model_training import train_model, get_feature_importance
from .model_evaluation import evaluate_model, analyze_predictions
from .visualizer import create_visualizations, plot_predictions_vs_actual
from .utils import setup_logging, ensure_directories, print_banner

__all__ = [
    'load_data',
    'validate_data',
    'engineer_features',
    'train_model',
    'get_feature_importance',
    'evaluate_model',
    'analyze_predictions',
    'create_visualizations',
    'plot_predictions_vs_actual',
    'setup_logging',
    'ensure_directories',
    'print_banner',
]
