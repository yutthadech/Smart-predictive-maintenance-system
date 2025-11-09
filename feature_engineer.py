"""
Feature Engineering Module
===========================

Functions for computing RUL and engineering features.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger('predictive_maintenance')


def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    output_dir: str = 'outputs'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete feature engineering pipeline.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        truth_df: Ground truth RUL DataFrame
        output_dir: Directory to save processed data
        
    Returns:
        Tuple of (train_processed, test_processed)
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Step 1: Compute RUL
    logger.info("Computing RUL for training data...")
    train_processed = compute_rul_training(train_df)
    
    logger.info("Computing RUL for test data...")
    test_processed = compute_rul_test(test_df, truth_df)
    
    # Step 2: Add rolling features
    logger.info("Adding rolling window features...")
    train_processed = add_rolling_features(train_processed, window_size=5)
    test_processed = add_rolling_features(test_processed, window_size=5)
    
    # Step 3: Add statistical features
    logger.info("Adding statistical aggregation features...")
    train_processed = add_statistical_features(train_processed)
    test_processed = add_statistical_features(test_processed)
    
    # Step 4: Export processed data
    logger.info("Exporting processed data...")
    train_path = os.path.join(output_dir, 'processed_train.csv')
    test_path = os.path.join(output_dir, 'processed_test.csv')
    
    os.makedirs(output_dir, exist_ok=True)
    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)
    
    logger.info(f" Feature engineering complete")
    logger.info(f"   Training features: {train_processed.shape[1]}")
    logger.info(f"   Test features:     {test_processed.shape[1]}")
    logger.info(f"   Saved to: {output_dir}/")
    
    return train_processed, test_processed


def compute_rul_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life for training data.
    
    Args:
        df: Training DataFrame
        
    Returns:
        DataFrame with RUL column added
    """
    df = df.copy()
    
    # Calculate max cycles per unit
    max_cycles = df.groupby('unit')['time'].max().reset_index()
    max_cycles.columns = ['unit', 'max_time']
    
    # Merge and compute RUL
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL'] = df['max_time'] - df['time']
    df = df.drop('max_time', axis=1)
    
    return df


def compute_rul_test(test_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life for test data.
    
    Args:
        test_df: Test DataFrame
        truth_df: Ground truth RUL DataFrame
        
    Returns:
        DataFrame with RUL column added
    """
    test_df = test_df.copy()
    
    # Get last cycle for each unit
    last_cycles = test_df.groupby('unit')['time'].max().reset_index()
    last_cycles.columns = ['unit', 'last_cycle']
    
    # Add ground truth RUL
    truth_df = truth_df.copy()
    truth_df['unit'] = range(1, len(truth_df) + 1)
    
    # Merge
    test_df = test_df.merge(last_cycles, on='unit', how='left')
    test_df = test_df.merge(truth_df[['unit', 'RUL']], on='unit', how='left', suffixes=('', '_truth'))
    
    # Compute RUL for each timestep
    test_df['RUL'] = test_df['RUL'] + (test_df['last_cycle'] - test_df['time'])
    test_df = test_df.drop('last_cycle', axis=1)
    
    return test_df


def add_rolling_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Add rolling window features for sensor data.
    
    Args:
        df: Input DataFrame
        window_size: Size of rolling window
        
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    # Select key sensors for rolling features
    sensor_cols = [f'sensor_{i}' for i in [2, 3, 4, 7, 11, 15]]
    
    for col in sensor_cols:
        if col in df.columns:
            # Rolling mean
            df[f'{col}_rolling_mean'] = df.groupby('unit')[col].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'{col}_rolling_std'] = df.groupby('unit')[col].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).std()
            ).fillna(0)
    
    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add statistical aggregation features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with statistical features added
    """
    df = df.copy()
    
    # Get all sensor columns
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    
    for col in sensor_cols:
        # Mean per unit
        unit_mean = df.groupby('unit')[col].transform('mean')
        df[f'{col}_unit_mean'] = unit_mean
        
        # Standard deviation per unit
        unit_std = df.groupby('unit')[col].transform('std').fillna(0)
        df[f'{col}_unit_std'] = unit_std
        
        # Max per unit
        unit_max = df.groupby('unit')[col].transform('max')
        df[f'{col}_unit_max'] = unit_max
        
        # Deviation from mean
        df[f'{col}_deviation'] = df[col] - unit_mean
    
    return df

