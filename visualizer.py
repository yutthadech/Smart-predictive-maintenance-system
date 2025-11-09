"""
Visualizer Module
=================

Functions for creating plots and visualizations.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

logger = logging.getLogger('predictive_maintenance')

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    # Fallback to default style if seaborn style not available
    plt.style.use('default')
sns.set_palette("husl")


def safe_plot_save(fig, filepath, dpi=300):
    """Safely save plot and clean up resources."""
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure
    except Exception as e:
        logger.warning(f"Could not save plot to {filepath}: {e}")
        plt.close(fig)
    finally:
        plt.close('all')  # Close all figures to free memory


def create_visualizations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = 'outputs'
) -> None:
    """
    Create comprehensive visualizations for EDA.
    
    Args:
        train_df: Processed training DataFrame
        test_df: Processed test DataFrame
        output_dir: Directory to save plots
    """
    logger.info("="*80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. RUL Distribution
    logger.info("Creating RUL distribution plot...")
    plot_rul_distribution(train_df, test_df, output_dir)
    
    # 2. Sensor Correlation Heatmap
    logger.info("Creating sensor correlation heatmap...")
    plot_sensor_correlation(train_df, output_dir)
    
    # 3. Degradation Patterns
    logger.info("Creating sensor degradation patterns...")
    plot_degradation_patterns(train_df, output_dir)
    
    # 4. Feature Importance (if available)
    logger.info("Creating feature importance plot...")
    plot_feature_importance(output_dir)
    
    # 5. Lifecycle Examples
    logger.info("Creating lifecycle examples...")
    plot_lifecycle_examples(train_df, output_dir, n_units=3)
    
    logger.info(f"\n All visualizations saved to: {output_dir}/")


def plot_rul_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
) -> None:
    """Plot RUL distribution for train and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training RUL
    axes[0].hist(train_df['RUL'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(train_df['RUL'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {train_df["RUL"].mean():.1f}')
    axes[0].set_xlabel('Remaining Useful Life (cycles)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Training Set RUL Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Test RUL
    axes[1].hist(test_df['RUL'], bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(test_df['RUL'].mean(), color='red', linestyle='--',
                    label=f'Mean: {test_df["RUL"].mean():.1f}')
    axes[1].set_xlabel('Remaining Useful Life (cycles)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Test Set RUL Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    safe_plot_save(fig, os.path.join(output_dir, 'rul_distribution.png'))


def plot_sensor_correlation(df: pd.DataFrame, output_dir: str) -> None:
    """Plot correlation heatmap for sensor data."""
    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and '_' not in col[7:]]
    
    if not sensor_cols:
        return
    
    # Calculate correlation
    corr = df[sensor_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title('Sensor Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    safe_plot_save(fig, os.path.join(output_dir, 'sensor_correlation.png'))


def plot_degradation_patterns(df: pd.DataFrame, output_dir: str, n_units: int = 5) -> None:
    """Plot sensor degradation patterns for sample units."""
    # Select key sensors
    key_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7']
    key_sensors = [s for s in key_sensors if s in df.columns]
    
    if not key_sensors:
        return
    
    # Select random units
    units = df['unit'].unique()[:n_units]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, sensor in enumerate(key_sensors):
        ax = axes[idx]
        
        for unit in units:
            unit_data = df[df['unit'] == unit].sort_values('time')
            ax.plot(unit_data['time'], unit_data[sensor], 
                   label=f'Unit {unit}', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Time (cycles)', fontsize=12)
        ax.set_ylabel('Sensor Value', fontsize=12)
        ax.set_title(f'{sensor.replace("_", " ").title()} Degradation', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    safe_plot_save(fig, os.path.join(output_dir, 'degradation_patterns.png'))


def plot_feature_importance(output_dir: str, top_n: int = 20) -> None:
    """Plot feature importance from saved CSV."""
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    
    if not os.path.exists(importance_path):
        return
    
    df = pd.read_csv(importance_path).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(df)), df['importance'], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    safe_plot_save(fig, os.path.join(output_dir, 'feature_importance.png'))


def plot_lifecycle_examples(df: pd.DataFrame, output_dir: str, n_units: int = 3) -> None:
    """Plot complete lifecycle for sample units."""
    units = df['unit'].unique()[:n_units]
    
    for unit in units:
        unit_data = df[df['unit'] == unit].sort_values('time')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot 1: Key sensors over time
        ax1 = axes[0]
        key_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7']
        key_sensors = [s for s in key_sensors if s in df.columns]
        
        for sensor in key_sensors:
            ax1.plot(unit_data['time'], unit_data[sensor], 
                    label=sensor.replace('_', ' ').title(), linewidth=2)
        
        ax1.set_xlabel('Time (cycles)', fontsize=12)
        ax1.set_ylabel('Sensor Value', fontsize=12)
        ax1.set_title(f'Unit {unit} - Sensor Readings Over Time', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        
        # Plot 2: RUL over time
        ax2 = axes[1]
        ax2.plot(unit_data['time'], unit_data['RUL'], 
                color='red', linewidth=3, label='RUL')
        ax2.fill_between(unit_data['time'], unit_data['RUL'], 
                         alpha=0.3, color='red')
        ax2.set_xlabel('Time (cycles)', fontsize=12)
        ax2.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
        ax2.set_title(f'Unit {unit} - RUL Degradation', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        safe_plot_save(fig, os.path.join(output_dir, f'lifecycle_unit_{unit}.png'))


def plot_predictions_vs_actual(output_dir: str = 'outputs') -> None:
    """Plot predictions vs actual RUL."""
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    
    if not os.path.exists(predictions_path):
        logger.warning("Predictions file not found, skipping plot")
        return
    
    df = pd.read_csv(predictions_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(df['actual_RUL'], df['predicted_RUL'], 
               alpha=0.5, s=20, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    max_val = max(df['actual_RUL'].max(), df['predicted_RUL'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual RUL (cycles)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted RUL (cycles)', fontsize=12, fontweight='bold')
    ax1.set_title('Predictions vs Actual RUL', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Error distribution
    ax2 = axes[1]
    errors = df['actual_RUL'] - df['predicted_RUL']
    ax2.hist(errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean Error: {errors.mean():.2f}')
    ax2.set_xlabel('Prediction Error (cycles)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    safe_plot_save(fig, os.path.join(output_dir, 'predictions_analysis.png'))
    
    logger.info(f" Predictions plot saved to: {os.path.join(output_dir, 'predictions_analysis.png')}")

