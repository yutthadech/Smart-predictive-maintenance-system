"""
Data Preprocessing Module
==========================

Functions for loading and validating NASA C-MAPSS dataset.
"""

import os
import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger('predictive_maintenance')


def load_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load NASA C-MAPSS FD001 dataset.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (train_df, test_df, truth_df)
        
    Raises:
        FileNotFoundError: If data files are missing
    """
    logger.info("Loading NASA C-MAPSS FD001 dataset...")
    
    # File paths
    train_path = os.path.join(data_dir, 'train_FD001.txt')
    test_path = os.path.join(data_dir, 'test_FD001.txt')
    rul_path = os.path.join(data_dir, 'RUL_FD001.txt')
    
    # Check if files exist
    for path in [train_path, test_path, rul_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
    
    # Load data
    train_df = pd.read_csv(train_path, sep=' ', header=None)
    test_df = pd.read_csv(test_path, sep=' ', header=None)
    truth_df = pd.read_csv(rul_path, sep=' ', header=None)
    
    # Drop empty columns
    train_df.dropna(axis=1, inplace=True)
    test_df.dropna(axis=1, inplace=True)
    truth_df.dropna(axis=1, inplace=True)
    
    # Rename columns
    cols = ['unit', 'time'] + \
           [f'op_setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]
    
    train_df.columns = cols
    test_df.columns = cols
    truth_df.columns = ['RUL']
    
    logger.info(f"Data loaded successfully")
    logger.info(f"  Training data: {train_df.shape}")
    logger.info(f"  Test data:     {test_df.shape}")
    logger.info(f"  Ground truth:  {truth_df.shape}")
    
    # Validate data
    validate_data(train_df, test_df, truth_df)
    
    return train_df, test_df, truth_df


def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame, truth_df: pd.DataFrame) -> None:
    """
    Validate loaded data.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        truth_df: Ground truth DataFrame
    """
    logger.info("Validating data quality...")
    
    # Check for missing values
    if train_df.isnull().any().any():
        logger.warning("Training data contains missing values")
    
    if test_df.isnull().any().any():
        logger.warning("Test data contains missing values")
    
    # Check number of units
    n_train_units = train_df['unit'].nunique()
    n_test_units = test_df['unit'].nunique()
    n_truth_units = len(truth_df)
    
    logger.info(f"  Training units: {n_train_units}")
    logger.info(f"  Test units:     {n_test_units}")
    logger.info(f"  Truth units:    {n_truth_units}")
    
    if n_test_units != n_truth_units:
        logger.warning(f"Mismatch: {n_test_units} test units but {n_truth_units} truth values")
    
    logger.info("Data validation complete")


def display_data_info(train_df: pd.DataFrame, test_df: pd.DataFrame, truth_df: pd.DataFrame) -> None:
    """
    Display dataset information.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        truth_df: Ground truth DataFrame
    """
    print("\nDATASET INFORMATION")
    print("=" * 80)
    
    print(f"\nDataset Shapes:")
    print(f"  Training:     {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"  Test:         {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
    print(f"  Ground Truth: {truth_df.shape[0]:,} values")
    
    print(f"\nNumber of Engines:")
    print(f"  Training set: {train_df['unit'].nunique()}")
    print(f"  Test set:     {test_df['unit'].nunique()}")
    
    print(f"\nCycles per Engine:")
    train_cycles = train_df.groupby('unit')['time'].max()
    test_cycles = test_df.groupby('unit')['time'].max()
    print(f"  Training - Avg: {train_cycles.mean():.1f}, Min: {train_cycles.min()}, Max: {train_cycles.max()}")
    print(f"  Test     - Avg: {test_cycles.mean():.1f}, Min: {test_cycles.min()}, Max: {test_cycles.max()}")
    
    print(f"\nSample Training Data (first 5 rows):")
    print(train_df.head())
    
    print(f"\nSample Ground Truth RUL (first 10 values):")
    print(truth_df.head(10))

