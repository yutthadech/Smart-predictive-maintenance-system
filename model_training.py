"""
Model Training Module
=====================

Functions for training Random Forest models.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger('predictive_maintenance')


def train_model(
    train_path: str,
    model_dir: str = 'models',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    test_size: float = 0.2
) -> dict:
    """
    Train Random Forest model for RUL prediction.
    
    Args:
        train_path: Path to processed training data
        model_dir: Directory to save trained model
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random state for reproducibility
        test_size: Validation split ratio
        
    Returns:
        Dictionary with training metrics
    """
    logger.info("="*80)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("="*80)
    
    # Load processed data
    logger.info(f"Loading training data from: {train_path}")
    df = pd.read_csv(train_path)
    
    # Prepare features and target
    X, y, feature_names = prepare_features_and_target(df)
    
    logger.info(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set:   {X_train.shape[0]:,} samples")
    logger.info(f"Validation set: {X_val.shape[0]:,} samples")
    
    # Train model
    logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_metrics = calculate_metrics(y_train, y_pred_train)
    val_metrics = calculate_metrics(y_val, y_pred_val)
    
    logger.info("Training complete!")
    logger.info(f"Training Metrics:")
    logger.info(f"  MAE:  {train_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {train_metrics['rmse']:.4f}")
    logger.info(f"  R2:   {train_metrics['r2']:.4f}")
    
    logger.info(f"Validation Metrics:")
    logger.info(f"  MAE:  {val_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {val_metrics['rmse']:.4f}")
    logger.info(f"  R2:   {val_metrics['r2']:.4f}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'random_forest_rul_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save feature names
    feature_path = os.path.join(model_dir, 'feature_names.txt')
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_names))
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model_path': model_path,
        'n_features': X.shape[1]
    }


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare features (X) and target (y) from DataFrame.
    
    Args:
        df: Input DataFrame with RUL column
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Drop non-feature columns
    drop_cols = ['unit', 'time', 'RUL']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X = df[feature_cols].values
    y = df['RUL'].values
    
    return X, y, feature_cols


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def get_feature_importance(model_dir: str = 'models', output_dir: str = 'outputs', top_n: int = 20) -> pd.DataFrame:
    """
    Extract and save feature importance from trained model.
    
    Args:
        model_dir: Directory containing trained model
        output_dir: Directory to save importance data
        top_n: Number of top features to display
        
    Returns:
        DataFrame with feature importance
    """
    logger.info("Analyzing feature importance...")
    
    # Load model
    model_path = os.path.join(model_dir, 'random_forest_rul_model.pkl')
    model = joblib.load(model_path)
    
    # Load feature names
    feature_path = os.path.join(model_dir, 'feature_names.txt')
    with open(feature_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Get importance
    importance = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    
    logger.info(f"Top {top_n} Most Important Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature'][:40]:<40} {row['importance']:.6f}")
    
    logger.info(f"Feature importance saved to: {importance_path}")
    
    return importance_df

