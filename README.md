# Smart Predictive Maintenance System

A production-ready machine learning pipeline for predicting Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS FD001 dataset.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Application Screenshots](#application-screenshots)
- [Model Performance](#model-performance)
- [Pipeline Output Visualizations](#pipeline-output-visualizations)

## Overview

This project implements an end-to-end machine learning pipeline for predictive maintenance of aircraft engines. Using sensor data from the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset, the system predicts the Remaining Useful Life (RUL) of engines, enabling proactive maintenance scheduling and preventing unexpected failures.

### Key Objectives

- **Predict RUL**: Forecast the remaining operational cycles before engine failure
- **Prevent Failures**: Enable proactive maintenance to avoid costly breakdowns
- **Optimize Costs**: Reduce maintenance costs through data-driven decisions
- **Production-Ready**: Clean, modular, and deployable code structure

## Features

### Data Processing
- Automated data loading and validation
- Handling of 21 sensor measurements and 3 operational settings
- Missing value detection and data quality checks

### Feature Engineering
- RUL computation for training and test datasets
- Rolling window features (mean, std) for key sensors
- Statistical aggregation features (mean, std, max, deviation per unit)
- **171 engineered features** for improved model performance

### Machine Learning
- **Random Forest Regressor** as the primary model
- Cross-validation and hyperparameter tuning
- Comprehensive evaluation metrics (MAE, RMSE, R², MAPE)
- Feature importance analysis

### Visualization
- RUL distribution analysis
- Sensor correlation heatmaps
- Degradation pattern visualization
- Predictions vs actual plots
- Residual analysis

### Interactive Dashboard
- **Streamlit web app** for model exploration
- Real-time prediction visualization
- Feature importance analysis
- Unit-specific performance tracking

## Project Structure

```
Smart-Predictive-Maintenance/
├── app/
│   └── main_app.py              # Streamlit dashboard
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_preprocessing.py     # Data loading and validation
│   ├── feature_engineer.py      # RUL computation and feature engineering
│   ├── model_training.py        # Random Forest training
│   ├── model_evaluation.py      # Model evaluation and metrics
│   ├── visualizer.py            # Plotting and visualization
│   └── utils.py                 # Logging and utilities
├── models/
│   ├── random_forest_rul_model.pkl   # Trained model (generated)
│   └── feature_names.txt             # Feature list (generated)
├── outputs/
│   ├── processed_train.csv           # Processed training data (generated)
│   ├── processed_test.csv            # Processed test data (generated)
│   ├── test_metrics.json             # Evaluation metrics (generated)
│   ├── predictions.csv               # Model predictions (generated)
│   ├── feature_importance.csv        # Feature importance (generated)
│   └── *.png                         # Visualizations (generated)
├── data/
│   ├── train_FD001.txt          # Training data (add your own)
│   ├── test_FD001.txt           # Test data (add your own)
│   └── RUL_FD001.txt            # Ground truth RUL (add your own)
├── main.py                      # Main pipeline entry point
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Meriem-Sakka/smart-predictive-maintenance.git
cd smart-predictive-maintenance
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Dataset

Download the NASA C-MAPSS FD001 dataset and place the files in the `data/` directory:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

**Dataset Source**: [NASA Prognostics Data Repository](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

## Usage

### Run the Complete Pipeline

Execute the end-to-end pipeline (data loading → feature engineering → training → evaluation):

```bash
python main.py
```

This will:
1. Load and validate the NASA C-MAPSS dataset
2. Engineer 171 features from raw sensor data
3. Train a Random Forest model
4. Evaluate on the test set
5. Generate visualizations and save results to `outputs/`

### Launch the Interactive Dashboard

Start the Streamlit web application:

```bash
streamlit run app/main_app.py
```

The dashboard will open in your browser at `http://localhost:8501` and provide:
- Model performance metrics
- Interactive predictions analysis
- Feature importance visualization
- Unit-specific RUL tracking

## Application Screenshots

### Overview Dashboard
![Overview Dashboard](screenshots/screenshot_overview.png)
*Interactive dashboard showing project overview, key performance metrics (MAE: 55.88, RMSE: 71.05, R²: -0.45), technical details, and 6 comprehensive analysis tabs*

### Data Insights - RUL Distribution
![RUL Distribution Analysis](screenshots/screenshot_rul_distribution.png)
*Comparative analysis of Remaining Useful Life distribution between training and test datasets with detailed summary statistics*

### Sensor Correlation Analysis
![Sensor Correlation Heatmap](screenshots/screenshot_correlation.png)
*Correlation heatmap revealing relationships between 21 sensors. Note: Sensors 1, 5, 10, 16, 18, 19 show zero variance in FD001 (single operating condition)*

### Predictions Analysis
![Predictions vs Actual](screenshots/screenshot_predictions.png)
*Scatter plot comparison of predicted vs actual RUL with error distribution analysis, showing model performance across all test units*

### Unit-Specific Analysis
![Unit Analysis](screenshots/screenshot_unit_analysis.png)
*Detailed RUL prediction tracking for individual engine units, comparing actual vs predicted values over operational cycles with error visualization*

### Maintenance Insights & Feature Importance
![Maintenance Insights](screenshots/screenshot_maintenance.png)
*Comprehensive maintenance dashboard showing color-coded alerts (🚨 Critical, ⚠️ Warning, ✅ Healthy) and feature importance analysis. Sensor 11 rolling mean deviation dominates with 68% importance, followed by Sensor 4 and 3 statistics*


## Model Performance

### Test Set Results

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | 55.88 cycles | Mean Absolute Error |
| **RMSE** | 71.05 cycles | Root Mean Squared Error |
| **R² Score** | -0.45 | Coefficient of Determination |
| **MAPE** | 35.11% | Mean Absolute Percentage Error |

> **Note**: The negative R² score (-0.45) indicates that the current model performs worse than simply predicting the mean RUL value. This suggests the need for model improvement through hyperparameter tuning, feature selection, or alternative algorithms.

### Performance Highlights

- ⚠️ **R² Score: -0.45**: Model performs below baseline (predicting mean)
- ✅ **Accuracy within 10 cycles**: 15.76% of predictions
- ✅ **Accuracy within 20 cycles**: 26.19% of predictions
- ⚠️ **High MAPE**: 35.11% indicates significant prediction errors
- 📊 **Feature Importance**: Sensor 11 rolling mean deviation dominates (68% importance)

## Pipeline Outputs

The pipeline automatically generates comprehensive outputs in the `outputs/` directory:

### Generated Files
- **Data**: `processed_train.csv`, `processed_test.csv` - Processed datasets with 171 features
- **Model**: `random_forest_rul_model.pkl` - Trained Random Forest model
- **Metrics**: `test_metrics.json` - Performance evaluation results
- **Predictions**: `predictions.csv` - Model predictions with error analysis
- **Analysis**: `feature_importance.csv` - Feature importance rankings
- **Visualizations**: Multiple PNG plots for data analysis and model evaluation

> **For interactive exploration and polished visualizations, see the [Application Screenshots](#-application-screenshots) section above**

## Technologies

### Core Libraries
- **Python 3.8+**: Programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms

### Visualization
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization

### Web Framework
- **Streamlit**: Interactive dashboard

### Model Persistence
- **Joblib**: Model serialization

## Future Enhancements

- [ ] **Model Improvement**: Hyperparameter tuning and feature selection to improve R² score
- [ ] **Advanced Algorithms**: Add LSTM/GRU deep learning models for better time-series prediction
- [ ] **Feature Engineering**: Explore additional domain-specific features and transformations
- [ ] **Cross-validation**: Implement proper time-series cross-validation
- [ ] **Ensemble Methods**: Combine multiple models for improved performance
- [ ] **Real-time API**: Implement prediction API for production deployment
- [ ] **Anomaly Detection**: Add unsupervised learning for fault detection
- [ ] **Multi-dataset Support**: Extend to FD002, FD003, FD004 datasets
- [ ] **Docker Containerization**: Package for easy deployment
- [ ] **Cloud Deployment**: AWS/Azure/GCP integration
- [ ] **Model Versioning**: MLflow integration for experiment tracking


