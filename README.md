# CuNi Thin Film Optimization - Recipe Prediction System

## Overview

This is a machine learning-based optimization system for predicting and recommending optimal recipes for CuNi thin film preparation. The system trains multiple regression models to predict the Single-crystal Ratio of Substrate (G) based on experimental parameters, then uses differential evolution to search for optimal recipes that maximize this property while maintaining hBN Domain Alignment (H) = 1.

**Key Features:**
- Automatic hyperparameter tuning using Grid Search with 5-fold cross-validation
- 6 different regression models: XGBoost, Gradient Boosting Regressor (GBR), Lasso, Random Forest, SVR, and MLP
- SHAP (SHapley Additive exPlanations) values for model interpretability
- t-SNE dimensionality reduction visualization
- Automatic recipe optimization using differential evolution
- Comprehensive visualization and data export

(1) System Requirements

# Operating System
- **Linux** or **Windows** with Python 3.9+

# Python Version
- **Python 3.9** or higher (recommended: Python 3.10+)

# Required Libraries and Versions

pandas >= 1.3.0           # Data manipulation and analysis
numpy >= 1.20.0           # Numerical computing
scikit-learn >= 1.0.0     # Machine learning algorithms
xgboost >= 1.5.0          # XGBoost gradient boosting
scipy >= 1.7.0            # Scientific computing (optimization, statistics)
matplotlib >= 3.4.0       # Data visualization
shap >= 0.40.0            # SHAP model interpretability
joblib >= 1.0.0           # Model serialization (optional)
openpyxl >= 3.6.0         # Excel file writing

(2) Installation Instructions

#Step 1: Create a Virtual Environment (Recommended)

# Using venv (built-in)
python3 -m venv cuni_env
source cuni_env/bin/activate  # On Windows: cuni_env\Scripts\activate

# Or using conda (alternative)
conda create -n cuni_env python=3.10
conda activate cuni_env

# Step 2: Install Required Packages

**Option A: Using pip**
pip install --upgrade pip
pip install pandas>=1.3.0 numpy>=1.20.0 scikit-learn>=1.0.0 xgboost>=1.5.0 scipy>=1.7.0 matplotlib>=3.4.0 shap>=0.40.0 openpyxl>=3.6.0

**Option B: Using conda**
conda install pandas numpy scikit-learn xgboost scipy matplotlib shap openpyxl

(3) Input Data Format

# CSV File Structure

The input CSV file must follow this exact format:

| Column | Position | Content | Data Type | Description |
|--------|----------|---------|-----------|-------------|
| A | 1 | Row Index | Integer | Sample identifier or row number |
| B | 2 | **Sputtering Order** | Numeric (0, 1, or 2) | Order of sputtering parameter |
| C | 3 | **Ni Content** | Numeric | Nickel content percentage or value |
| D | 4 | **Sputtering Power** | Numeric | Applied sputtering power |
| E | 5 | **Temperature** | Numeric | Process temperature |
| F | 6 | **H2 Flow Rate** | Numeric | Hydrogen gas flow rate |
| G | 7 | **Single-crystal Ratio of Substrate** | Numeric | Target output variable (continuous) |
| H | 8 | **hBN Domain Alignment** | Integer (1 or 2) | Categorical target variable |

### Example CSV Structure (first few rows)

```csv
Index,Sputtering_Order,Ni_Content,Sputtering_Power,Temperature,H2_Flow_Rate,Single_crystal_Ratio,hBN_Alignment
1,0,10,50,900,40,0.75,1
2,1,20,100,950,100,0.82,1
3,2,30,150,1000,500,0.88,1
4,0,40,200,1050,1000,0.90,2
...
```

(4)## Usage Guide

#### Step 1: Prepare Your Data

Place your CSV data file in the same directory as `prediction.py`:

```
01_CuNi-opt/
├── prediction.py
├── your_data.csv          # Your experimental data
└── README.md
```

#### Step 2: Configure Settings (Optional)

Edit the "User Settings" section in `prediction.py`:

```python
# If you have only one CSV file, leave as None (auto-detection)
CSV_FILE = None
# If you have multiple CSVs, specify the filename:
# CSV_FILE = "your_data.csv"

# Optimization objective:
# "max" - maximize Single-crystal Ratio of Substrate
# "min" - minimize Single-crystal Ratio of Substrate
OBJECTIVE = "max"

# Random seed for reproducibility (default: 42)
RANDOM_STATE = 42

# Output directory name (will be created in the script directory)
OUTPUT_DIR_NAME = "prediction_results"
```

#### Step 3: Run the Script

```bash
# Navigate to the script directory
cd path/to/01_CuNi-opt

# Run the script
python prediction.py
```

#### Step 4: Check the Results

Results will be saved in the `prediction_results/` directory (or custom `OUTPUT_DIR_NAME`):

```
prediction_results/
├── train_test_datasets.xlsx              # Training and test sets
├── model_comparison_G.csv                # Model performance comparison
├── optimal_recipes.csv                   # 3 optimized recipes
├── test_predictions_G.csv                # Test set predictions
├── test_predictions_G.png                # Prediction diagonal plot
├── feature_importance_G_GBR.png          # Feature importance visualization
├── feature_importance_G_GBR.xlsx         # Feature importance data
├── dimensionality_reduction_tSNE_tSNE1_tSNE2.png  # t-SNE visualization
├── dimensionality_reduction_coordinates.xlsx      # t-SNE coordinates
├── shap_values_G_<model_name>.png        # SHAP summary plots
└── shap_values_G_<model_name>.xlsx       # SHAP values data
```

(5) Output Files Detailed Description

(5.1) `train_test_datasets.xlsx`
- **Format:** Excel workbook with 2 sheets
- **Sheets:**
  - `Training Set`: 80% of data (rows with all input and output features)
  - `Test Set`: 20% of data (reserved for final model evaluation)
- **Columns:** Sputtering Order, Ni Content, Sputtering Power, Temperature, H2 Flow Rate, Single-crystal Ratio, hBN Domain Alignment

(5.2) `optimal_recipes.csv`
- **Content:** Top 3 optimized recipes found by differential evolution
- **Columns:**
  - `Recipe_ID`: Recipe number (1-3)
  - `Sputtering Order`: Sputtering order value
  - `Ni Content`: Nickel content
  - `Sputtering Power`: Sputtering power
  - `Temperature`: Process temperature
  - `H2 Flow Rate`: Hydrogen flow rate
  - `Predicted_G`: Predicted Single-crystal Ratio value
  - `Predicted_H`: Predicted hBN Domain Alignment (always 1)
  - `Objective`: Optimization objective description

(5.3) `model_comparison_G.csv`
- **Content:** Comparison of all 6 regression models
- **Columns:**
  - `Model`: Model name (XGBoost, GBR, Lasso, RF, SVR, MLP)
  - `Test_RMSE`: Root Mean Squared Error on test set
  - `Test_R2`: R² score on test set (ranges from 0 to 1, higher is better)
- **Sorted by:** Test RMSE (ascending) - best models first


(5.4) `test_predictions_G.csv`
- **Content:** Actual vs. predicted values for test set
- **Columns:**
  - `Observed_G`: Actual Single-crystal Ratio values from test set
  - `<ModelName>_Test_Predicted_G`: Predicted values from each model

#### `test_predictions_G.png`
- **Type:** Scatter plot visualization
- **Content:** Diagonal plots showing actual vs. predicted values for each model
- **Interpretation:** 
  - Points on the red diagonal line = perfect predictions
  - Points above/below = over/under predictions
  - RMSE and R² displayed for each model

(5.5) `feature_importance_G_GBR.xlsx`
- **Content:** Numerical values of feature importance
- **Columns:**
  - `Feature`: Feature name
  - `Importance`: Importance score (0 to 1)
- **Sorted by:** Importance (descending)

#### `feature_importance_G_GBR.png`
- **Type:** Horizontal bar chart
- **Content:** Importance of each input feature (Sputtering Order, Ni Content, etc.)
- **Model:** Gradient Boosting Regressor (typically best performer)
- **Interpretation:** Longer bars = more important features

(5.6)`shap_values_G_<model_name>.xlsx`
- **Content:** Raw SHAP values for each sample and feature
- **Columns:**
  - `Sample_Index`: Sample identifier
  - `<Feature>`: SHAP values for each feature
  - `<Feature>_value`: Original feature values
  
####`shap_values_G_<model_name>.png`
- **Type:** SHAP summary plot
- **Content:** Shows how each feature contributes to predictions
- **Interpretation:**
  - Red dots = high feature values → positive/negative contribution
  - Blue dots = low feature values → opposite contribution
  - Horizontal position = magnitude of contribution

(5.7) `dimensionality_reduction_coordinates.xlsx`
- **Format:** Excel spreadsheet
- **Content:** t-SNE coordinates for visualization and analysis
- **Columns:**
  - `tSNE1`: First t-SNE dimension
  - `tSNE2`: Second t-SNE dimension
  - `G`: Single-crystal Ratio value
  - `Data_Type`: "Training Set" or "Optimized Recipe"
  - (Optional) `Recipe_ID`: ID for optimized recipes
  
#### `dimensionality_reduction_tSNE_tSNE1_tSNE2.png`
- **Type:** 2D scatter plot
- **Content:** t-SNE visualization of high-dimensional data
- **Color:** Colored by Single-crystal Ratio values (viridis colormap)
- **Markers:**
  - Blue to yellow dots = training set samples (colored by G value)
  - Red stars = optimized recipe points (labeled 1, 2, 3)
- **Interpretation:** Samples close in 2D space are similar in high-dimensional space

