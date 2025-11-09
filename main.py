"""
NASA C-MAPSS Predictive Maintenance Pipeline
=============================================

Main entry point for the end-to-end RUL prediction pipeline.

Usage:
    python main.py

Author: AI Engineer
Date: October 2025
Version: 2.0.0
"""

import os
import sys
from datetime import datetime

# Configure matplotlib backend early to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline modules
from src import (
    load_data,
    engineer_features,
    train_model,
    evaluate_model,
    create_visualizations,
    setup_logging,
    ensure_directories,
    print_banner,
    get_feature_importance,
    analyze_predictions,
    plot_predictions_vs_actual
)

from src.utils import print_section, print_summary
from src.data_preprocessing import display_data_info


def main():
    """Execute the complete predictive maintenance pipeline."""
    
    # Record start time
    start_time = datetime.now()
    
    # Print banner
    print_banner()
    
    # Setup directories
    ensure_directories('data', 'models', 'outputs')
    
    # Setup logging
    log_file = os.path.join('outputs', 'pipeline.log')
    logger = setup_logging(log_file=log_file)
    logger.info("="*80)
    logger.info("NASA C-MAPSS PREDICTIVE MAINTENANCE PIPELINE STARTED")
    logger.info("="*80)
    
    try:
        # ====================================================================
        # STEP 1: Load Data
        # ====================================================================
        print_section("STEP 1/6: LOADING DATA")
        
        train_df, test_df, truth_df = load_data(data_dir='data')
        display_data_info(train_df, test_df, truth_df)
        
        logger.info("Step 1 completed: Data loaded successfully")
        
        # ====================================================================
        # STEP 2: Feature Engineering
        # ====================================================================
        print_section("STEP 2/6: FEATURE ENGINEERING")
        
        train_processed, test_processed = engineer_features(
            train_df, test_df, truth_df, output_dir='outputs'
        )
        
        logger.info(f"Step 2 completed: {train_processed.shape[1]} features engineered")
        
        # ====================================================================
        # STEP 3: Create Visualizations
        # ====================================================================
        print_section("STEP 3/6: CREATING VISUALIZATIONS")
        
        create_visualizations(train_processed, test_processed, output_dir='outputs')
        
        logger.info("Step 3 completed: Visualizations created")
        
        # ====================================================================
        # STEP 4: Train Model
        # ====================================================================
        print_section("STEP 4/6: TRAINING RANDOM FOREST MODEL")
        
        train_metrics = train_model(
            train_path='outputs/processed_train.csv',
            model_dir='models',
            n_estimators=100,
            max_depth=None,
            random_state=42,
            test_size=0.2
        )
        
        logger.info("Step 4 completed: Model trained successfully")
        
        # ====================================================================
        # STEP 5: Evaluate Model
        # ====================================================================
        print_section("STEP 5/6: EVALUATING MODEL ON TEST SET")
        
        test_metrics = evaluate_model(
            test_path='outputs/processed_test.csv',
            model_dir='models',
            output_dir='outputs'
        )
        
        logger.info("Step 5 completed: Model evaluated on test set")
        
        # ====================================================================
        # STEP 6: Feature Importance Analysis
        # ====================================================================
        print_section("STEP 6/6: ANALYZING FEATURE IMPORTANCE")
        
        importance_df = get_feature_importance(
            model_dir='models',
            output_dir='outputs',
            top_n=20
        )
        
        logger.info("Step 6 completed: Feature importance analyzed")
        
        # ====================================================================
        # Additional Analysis
        # ====================================================================
        analyze_predictions(output_dir='outputs')
        plot_predictions_vs_actual(output_dir='outputs')
        
        # ====================================================================
        # Print Summary
        # ====================================================================
        print_summary(start_time, test_metrics)
        
        logger.info("="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nError: {e}")
        print("\nPlease ensure the following files exist in the 'data/' directory:")
        print("  - train_FD001.txt")
        print("  - test_FD001.txt")
        print("  - RUL_FD001.txt")
        return 1
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
