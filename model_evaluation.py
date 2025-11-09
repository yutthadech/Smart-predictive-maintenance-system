"""
Model Evaluation Module
=======================

Functions for evaluating trained models on test data.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger('predictive_maintenance')


def evaluate_model(
    test_path: str,
    model_dir: str = 'models',
    output_dir: str = 'outputs'
) -> dict:
    """
    Evaluate trained model on test data.
    
    Args:
        test_path: Path to processed test data
        model_dir: Directory containing trained model
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with test metrics
    """
    logger.info("="*80)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("="*80)
    
    # Load model
    model_path = os.path.join(model_dir, 'random_forest_rul_model.pkl')
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Load test data
    logger.info(f"Loading test data from: {test_path}")
    df = pd.read_csv(test_path)
    
    # Prepare features and target
    drop_cols = ['unit', 'time', 'RUL']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X_test = df[feature_cols].values
    y_test = df['RUL'].values
    
    logger.info(f"Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_metrics = calculate_test_metrics(y_test, y_pred)
    
    logger.info("Evaluation complete!")
    logger.info(f"Test Set Performance:")
    logger.info(f"  MAE:   {test_metrics['mae']:.4f} cycles")
    logger.info(f"  RMSE:  {test_metrics['rmse']:.4f} cycles")
    logger.info(f"  R2:    {test_metrics['r2']:.4f}")
    logger.info(f"  MAPE:  {test_metrics['mape']:.2f}%")
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'unit': df['unit'],
        'time': df['time'],
        'actual_RUL': y_test,
        'predicted_RUL': y_pred,
        'error': y_test - y_pred,
        'abs_error': np.abs(y_test - y_pred)
    })
    
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    return test_metrics


def calculate_test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive test metrics.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        
    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Accuracy within threshold
    threshold_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
    threshold_20 = np.mean(np.abs(y_true - y_pred) <= 20) * 100
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'accuracy_within_10_cycles': float(threshold_10),
        'accuracy_within_20_cycles': float(threshold_20)
    }


def analyze_predictions(output_dir: str = 'outputs') -> None:
    """
    Analyze prediction errors and print insights.
    
    Args:
        output_dir: Directory containing predictions
    """
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    
    if not os.path.exists(predictions_path):
        logger.warning("Predictions file not found")
        return
    
    df = pd.read_csv(predictions_path)
    
    logger.info("Prediction Analysis:")
    
    # Overall stats
    logger.info(f"  Overall Statistics:")
    logger.info(f"  Mean Error:      {df['error'].mean():.4f}")
    logger.info(f"  Std Error:       {df['error'].std():.4f}")
    logger.info(f"  Max Overpredict: {df['error'].max():.4f}")
    logger.info(f"  Max Underpredict:{df['error'].min():.4f}")
    
    # Per-unit analysis
    unit_errors = df.groupby('unit')['abs_error'].mean().sort_values(ascending=False)
    
    logger.info(f"  Top 5 Units with Highest Average Error:")
    for idx, (unit, error) in enumerate(unit_errors.head(5).items(), 1):
        logger.info(f"  {idx}. Unit {unit}: {error:.2f} cycles")
    
    logger.info(f"  Top 5 Units with Lowest Average Error:")
    for idx, (unit, error) in enumerate(unit_errors.tail(5).items(), 1):
        logger.info(f"  {idx}. Unit {unit}: {error:.2f} cycles")

