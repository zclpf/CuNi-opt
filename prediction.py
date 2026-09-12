"""
Overall workflow:

1. Auto-find CSV data file in script directory
2. Read data by Excel column position: B-F columns are input features, G column is output
3. Grid search hyperparameters for 6 regression models, rank by cross-validation RMSE
4. Search within historical data range for recipe that maximizes (or minimizes) Single-crystal Ratio of Substrate, recommend from each model
5. Save models, comparison results, recommended recipes and predictions to output directory

Only modify the “User Settings” section below to run. No need to change function code.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display windows
import shap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# User Settings
# ============================================================

# If current directory has only one CSV, keep as None
# If there are multiple CSVs, specify filename, e.g.:
# CSV_FILE = "data.csv"
CSV_FILE = None

# "max": Find recipe that maximizes Single-crystal Ratio of Substrate
# "min": Find recipe that minimizes Single-crystal Ratio of Substrate
OBJECTIVE = "max"

# Random seed for reproducibility
# Affects cross-validation splits, tree model randomness, and differential evolution initial population
RANDOM_STATE = 42

# Output directory
# Created relative to prediction.py location, auto-created if not exists
OUTPUT_DIR_NAME = "prediction_results"

# Target variable name mappings
TARGET_NAMES = {
    "G": "Single-crystal Ratio of Substrate",
    "H": "hBN Domain Alignment"
}

FEATURE_NAMES = {
    "Sputtering Order": "Sputtering Order",
    "Ni Content": "Ni Content",
    "Sputtering Power": "Sputtering Power",
    "Temperature": "Temperature",
    "H2 Flow Rate": "H2 Flow Rate"
}


def find_csv_file(project_dir: Path) -> Path:
    """Auto-find CSV file in current directory."""

    # If filename specified, use directly without auto-search
    if CSV_FILE:
        csv_path = project_dir / CSV_FILE

        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find specified CSV file: {csv_path}")

        return csv_path

    # Sort to ensure same file selected on each run with multiple files
    csv_files = sorted(project_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in directory: {project_dir}\n"
            "Please place CSV file in prediction.py directory."
        )

    if len(csv_files) > 1:
        print("Warning: Multiple CSV files found in current directory:")
        for item in csv_files:
            print(f"  - {item.name}")

        print(f"CSV_FILE not specified, will read: {csv_files[0].name}")
        print("To read other file, modify CSV_FILE at top of code.")

    return csv_files[0]


def read_csv_with_encoding(csv_path: Path) -> pd.DataFrame:
    """
    Try reading CSV with common encodings.

    header=None:
        Don't use any row as header.

    skiprows=2:
        Skip first two rows of CSV, start reading from row 3.
    """

    # Try in order of commonality. Chinese Excel exports often use gb18030/gbk
    encodings = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
    ]

    # Record failure reason for each encoding, report all if all fail
    errors = []

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                csv_path,
                header=None,
                skiprows=2,
                encoding=encoding,
            )

            print(f"CSV encoding: {encoding}")
            return dataframe

        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")

    raise RuntimeError(
        "Unable to determine CSV file encoding.\n" + "\n".join(errors)
    )


def load_training_data(
    csv_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load training data from CSV file.

    Read data by Excel column position:
    B-F columns: input features (Sputtering Order, Ni Content, Sputtering Power, Temperature, H2 Flow Rate)
    G column: Single-crystal Ratio of Substrate (continuous, for regression)
    H column: hBN Domain Alignment (categorical, value 1 or 2)
    Starting from row 3: data area
    """

    dataframe = read_csv_with_encoding(csv_path)

    # Insufficient columns indicate file structure mismatch. Raise error early to avoid reading wrong columns.
    if dataframe.shape[1] < 8:
        raise ValueError(
            f"CSV has only {dataframe.shape[1]} columns, "
            "but at least 8 columns (A-H) are required."
        )

    # pandas uses 0-based indexing:
    # B-F corresponds to positions 1:6
    # G corresponds to position 6
    # H corresponds to position 7
    x = dataframe.iloc[:, 1:6].copy()
    y_g = dataframe.iloc[:, 6].copy()
    y_h = dataframe.iloc[:, 7].copy()

    # Since header=None, column names are currently numeric. Rename to meaningful names for clarity.
    x.columns = ["Sputtering Order", "Ni Content", "Sputtering Power", "Temperature", "H2 Flow Rate"]
    y_g.name = "Single-crystal Ratio of Substrate"
    y_h.name = "hBN Domain Alignment"

    # Convert text to numeric values, converting non-numeric content to NaN.
    for column in x.columns:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    y_g = pd.to_numeric(y_g, errors="coerce")
    y_h = pd.to_numeric(y_h, errors="coerce")

    # Rows with missing G or H output cannot be used for supervised learning and are removed.
    valid_target = y_g.notna() & y_h.notna()
    x = x.loc[valid_target].reset_index(drop=True)
    y_g = y_g.loc[valid_target].reset_index(drop=True)
    y_h = y_h.loc[valid_target].reset_index(drop=True)

    # Check if input features have completely empty columns.
    # Such columns cannot be imputed and indicate either wrong column position or missing data.
    all_missing_columns = [
        column
        for column in x.columns
        if x[column].isna().all()
    ]

    if all_missing_columns:
        raise ValueError(
            "The following input columns have no valid numeric values: "
            + ", ".join(all_missing_columns)
        )

    # Too few samples make cross-validation meaningless.
    if len(x) < 5:
        raise ValueError(
            f"Valid sample count is only {len(x)}, at least 5 samples are required."
        )

    print("\nData loading completed")
    print(f"Valid sample count: {len(x)}")
    print(f"Input variables: {list(x.columns)}")
    print(f"Output variables: {y_g.name}, {y_h.name}")
    print("\nInput variable statistics:")
    print(x.describe().to_string())
    print(f"\nOutput variable {y_g.name} statistics:")
    print(y_g.describe().to_string())
    print(f"\nOutput variable {y_h.name} statistics:")
    print(y_h.describe().to_string())

    return x, y_g, y_h


def create_models_and_parameters():
    """
    Create models and hyperparameters to search.

    Returns a dictionary with model names as keys and (Pipeline, parameter grid) as values.

    Each model is wrapped in a Pipeline so that imputation and standardization are fit only
    on training folds, avoiding information leakage from validation folds into preprocessing.

    Parameter names with "model__" prefix point to the "model" step in the Pipeline.
    """

    models = {
        # Gradient boosting tree, usually performs best on small tabular datasets.
        "XGBoost": (
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [200, 500],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3, 5],
                "model__subsample": [0.8, 1.0],
            },
        ),

        # sklearn's built-in gradient boosting as a control for XGBoost.
        "GBR": (
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "model",
                        GradientBoostingRegressor(
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [1, 2],
            },
        ),

        # Linear regression with L1 regularization, serves as baseline and auto-selects useful features.
        # Regularization is scale-sensitive, so standardization is required.
        "Lasso": (
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        Lasso(
                            max_iter=100000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__alpha": [
                    0.0001,
                    0.001,
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],
            },
        ),

        # Random forest, low variance and insensitive to hyperparameters.
        "RF": (
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "model",
                        RandomForestRegressor(
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 1.0],
            },
        ),

        # RBF kernel support vector regression, suitable for strong non-linearity with small sample size.
        # Kernel function is distance-based, so standardization is required.
        "SVR": (
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        SVR(kernel="rbf"),
                    ),
                ]
            ),
            {
                "model__C": [1.0, 10.0, 100.0],
                "model__epsilon": [0.01, 0.05, 0.1],
                "model__gamma": ["scale", 0.01, 0.1],
            },
        ),

        # Deep neural network (multi-layer perceptron), suitable for fitting complex non-linear relationships.
        # Requires standardized inputs to accelerate convergence.
        "MLP": (
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        MLPRegressor(
                            random_state=RANDOM_STATE,
                            max_iter=5000,
                            early_stopping=True,
                            validation_fraction=0.2,
                            n_iter_no_change=50,
                            tol=1e-4,
                        ),
                    ),
                ]
            ),
            {
                "model__hidden_layer_sizes": [
                    (50,),
                    (100,),
                    (50, 50),
                ],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [0.001, 0.01, 0.1],
                "model__learning_rate": ["constant", "adaptive"],
                "model__learning_rate_init": [0.001, 0.01],
            },
        ),
    }

    return models


def train_and_compare_models(
    x: pd.DataFrame,
    y_g: pd.Series,
    y_h: pd.Series,
):
    """
    Split data with 80/20 ratio, train models on training set, evaluate on test set.

    Train two groups of models:
    - One group to predict Single-crystal Ratio of Substrate (regression)
    - One group to predict hBN Domain Alignment (classification)

    Returns (Single-crystal Ratio of Substrate model comparison table, best model name, best model, all models,
    test set predictions, best hBN Domain Alignment model, predictions, training set, test set).
    """

    # 80/20 split, maintaining correspondence between y_g and y_h
    x_train, x_test, y_g_train, y_g_test, y_h_train, y_h_test = train_test_split(
        x, y_g, y_h, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\nDataset split:")
    print(f"Training set samples: {len(x_train)} (80%)")
    print(f"Test set samples: {len(x_test)} (20%)")

    # Auto-reduce fold count for small samples.
    number_of_splits = min(5, len(x_train))

    # All models share the same cv object to ensure consistent splitting and fair comparison.
    cv = KFold(
        n_splits=number_of_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ========== Train Single-crystal Ratio of Substrate prediction model ==========
    print("\n" + "=" * 60)
    print(f"Train {TARGET_NAMES['G']} prediction model (regression)")
    print("=" * 60)

    models = create_models_and_parameters()

    comparison_g_records = []
    best_estimators_g = {}
    test_predictions_g = {}

    for model_name, (pipeline, parameter_grid) in models.items():
        print("\n" + "=" * 60)
        print(f"Training and tuning: {model_name}")
        print("=" * 60)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        # Grid search and cross-validation only on training set
        grid_search.fit(x_train, y_g_train)

        best_estimator = grid_search.best_estimator_
        best_estimators_g[model_name] = best_estimator

        # Predict on test set
        predictions = best_estimator.predict(x_test)
        test_predictions_g[model_name] = predictions

        # Calculate test set metrics
        rmse = float(
            np.sqrt(mean_squared_error(y_g_test, predictions))
        )
        mae = float(mean_absolute_error(y_g_test, predictions))
        r2 = float(r2_score(y_g_test, predictions))

        comparison_g_records.append(
            {
                "Model": model_name,
                "Test_RMSE": rmse,
                "Test_MAE": mae,
                "Test_R2": r2,
                "GridSearch_Best_RMSE": -grid_search.best_score_,
                "Best_Parameters": json.dumps(
                    grid_search.best_params_,
                    ensure_ascii=False,
                ),
            }
        )

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test set RMSE: {rmse:.8g}")
        print(f"Test set MAE: {mae:.8g}")
        print(f"Test set R2: {r2:.8g}")

    comparison_g = pd.DataFrame(comparison_g_records)
    comparison_g = comparison_g.sort_values(
        by=["Test_RMSE", "Test_MAE"],
        ascending=True,
    ).reset_index(drop=True)

    best_g_model_name = str(comparison_g.iloc[0]["Model"])
    best_g_model = best_estimators_g[best_g_model_name]

    # Refit best Single-crystal Ratio of Substrate model using entire training set.
    best_g_model.fit(x_train, y_g_train)

    # ========== Train hBN Domain Alignment prediction model (classification) ==========
    print("\n" + "=" * 60)
    print(f"Train {TARGET_NAMES['H']} prediction model (classification)")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # Simplified classification model, train only random forest
    h_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )

    h_param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 5, 10],
    }

    h_grid_search = GridSearchCV(
        estimator=h_pipeline,
        param_grid=h_param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    h_grid_search.fit(x_train, y_h_train)
    best_h_model = h_grid_search.best_estimator_

    h_predictions = best_h_model.predict(x_test)
    h_accuracy = accuracy_score(y_h_test, h_predictions)

    print(f"{TARGET_NAMES['H']} model best parameters: {h_grid_search.best_params_}")
    print(f"{TARGET_NAMES['H']} model test set accuracy: {h_accuracy:.4f}")
    print(f"\n{TARGET_NAMES['H']} model classification report:")
    print(classification_report(y_h_test, h_predictions))

    # Refit best hBN Domain Alignment model using entire training set
    best_h_model.fit(x_train, y_h_train)

    return (
        comparison_g,
        best_g_model_name,
        best_g_model,
        best_estimators_g,
        test_predictions_g,
        best_h_model,
        h_predictions,
        x_train,
        x_test,
        y_g_train,
        y_g_test,
        y_h_train,
        y_h_test,
    )


def optimize_recipe(
    model_g,
    model_h,
    x: pd.DataFrame,
    seed_offset: int = 0,
) -> tuple[np.ndarray, float, int]:
    """
    Search for optimal recipe within historical min/max of each variable.

    Optimization objective: maximize Single-crystal Ratio of Substrate while hBN Domain Alignment = 1.

    Restricting to historical ranges avoids pushing the model into extrapolation regions
    where predictions are unreliable.

    Special constraint: Sputtering Order (first column) can only take integer values 0, 1, 2.

    Parameters:
        model_g: trained Single-crystal Ratio of Substrate prediction model (regression)
        model_h: trained hBN Domain Alignment prediction model (classification)
        x: input data (used to determine search range)
        seed_offset: random seed offset to generate different recipes

    Returns (array of five variable values, corresponding predicted Single-crystal Ratio of Substrate, corresponding predicted hBN Domain Alignment).
    """

    lower_bounds = x.min(skipna=True).to_numpy(dtype=float)
    upper_bounds = x.max(skipna=True).to_numpy(dtype=float)

    # Force Sputtering Order (index 0) bounds to [0, 2]
    lower_bounds[0] = 0.0
    upper_bounds[0] = 2.0

    # Variables with equal bounds are constants in the data and don't need optimization.
    variable_indices = [
        index
        for index, (lower, upper) in enumerate(
            zip(lower_bounds, upper_bounds)
        )
        if not np.isclose(lower, upper)
    ]

    # Constant variables are fixed to their unique value (the lower bound at this point).
    fixed_values = lower_bounds.copy()

    # If all variables are constant, no optimization is needed.
    if not variable_indices:
        candidate = pd.DataFrame(
            [fixed_values],
            columns=x.columns,
        )
        predicted_g = float(model_g.predict(candidate)[0])
        predicted_h = int(model_h.predict(candidate)[0])
        return (fixed_values, predicted_g, predicted_h)

    optimization_bounds = [
        (
            float(lower_bounds[index]),
            float(upper_bounds[index]),
        )
        for index in variable_indices
    ]

    def make_full_recipe(variable_values):
        """Fill variable values back into complete five-variable recipe.

        For Sputtering Order (index 0), round its value to nearest integer {0, 1, 2}.
        """

        full_recipe = fixed_values.copy()

        for index, value in zip(
            variable_indices,
            variable_values,
        ):
            # If Sputtering Order (index 0), round to nearest integer and clip to [0, 2]
            if index == 0:
                value = float(np.clip(np.round(value), 0, 2))

            full_recipe[index] = value

        return full_recipe

    def objective_function(variable_values):
        """
        Objective function for differential evolution, always minimizes.

        Optimization objectives:
        1. hBN Domain Alignment must be predicted as 1
        2. Given hBN Domain Alignment=1, maximize Single-crystal Ratio of Substrate

        Strategy:
        - If hBN Domain Alignment is not 1, apply large penalty
        - If hBN Domain Alignment is 1, return -Single-crystal Ratio of Substrate (to maximize)
        """

        full_recipe = make_full_recipe(variable_values)

        # Convert to DataFrame while preserving column names
        # to match the input format seen during Pipeline training.
        candidate = pd.DataFrame(
            [full_recipe],
            columns=x.columns,
        )

        predicted_g = float(model_g.predict(candidate)[0])
        predicted_h = int(model_h.predict(candidate)[0])

        # If hBN Domain Alignment is not 1, return large penalty
        if predicted_h != 1:
            return 1e10  # large penalty

        # When hBN Domain Alignment is 1, maximize Single-crystal Ratio of Substrate
        # Return -Single-crystal Ratio of Substrate (since differential_evolution minimizes)
        return -predicted_g

    # Differential evolution is global optimization not relying on gradients,
    # suitable for tree models with step-like, non-differentiable objectives.
    # polish=True applies local optimization refinement after convergence.
    optimization_result = differential_evolution(
        objective_function,
        bounds=optimization_bounds,
        strategy="best1bin",
        maxiter=1000,
        popsize=20,
        tol=1e-8,
        polish=True,
        seed=RANDOM_STATE + seed_offset,
        workers=1,
    )

    best_recipe = make_full_recipe(optimization_result.x)

    best_candidate = pd.DataFrame(
        [best_recipe],
        columns=x.columns,
    )

    # Predict again to get true G and H values
    best_predicted_g = float(model_g.predict(best_candidate)[0])
    best_predicted_h = int(model_h.predict(best_candidate)[0])

    return (best_recipe, best_predicted_g, best_predicted_h)


def plot_prediction_diagonal(
    output_dir: Path,
    y_test: pd.Series,
    test_predictions: dict,
    comparison: pd.DataFrame,
):
    """
    Plot diagonal (actual vs predicted) plots for each G model.

    Two-row layout with one subplot per model, showing test set prediction performance.
    """

    num_models = len(test_predictions)
    # Changed to two-row layout, calculate column count per row
    ncols = (num_models + 1) // 2  # round up
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # Flatten axes to 1D array for uniform processing
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot in order of comparison (best model first)
    for idx, model_name in enumerate(comparison["Model"]):
        ax = axes[idx]
        predictions = test_predictions[model_name]

        # Draw scatter plot
        ax.scatter(y_test, predictions, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

        # Draw diagonal line (ideal prediction line)
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

        # Get metrics for this model
        model_metrics = comparison[comparison["Model"] == model_name].iloc[0]
        rmse = model_metrics["Test_RMSE"]
        r2 = model_metrics["Test_R2"]

        # Set title and labels
        ax.set_title(f'{model_name}\nRMSE={rmse:.4g}, R2={r2:.4g}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Actual {TARGET_NAMES["G"]}', fontsize=11)
        ax.set_ylabel(f'Predicted {TARGET_NAMES["G"]}', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Set same coordinate axis range so diagonal is 45 degrees
        ax.set_aspect('equal', adjustable='box')

    # Hide extra subplots
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "test_predictions_G.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{TARGET_NAMES['G']} diagonal plot saved: {output_dir / 'test_predictions_G.png'}")


def plot_i_prediction_diagonal(
    output_dir: Path,
    y_i_test: pd.Series,
    test_predictions_i: dict,
    comparison_i: pd.DataFrame,
):
    """
    This function has been removed as I column data is no longer processed.
    """
    pass


def plot_h_prediction(
    output_dir: Path,
    y_h_test: pd.Series,
    h_predictions: np.ndarray,
):
    """
    Plot hBN Domain Alignment model prediction comparison.

    Uses scatter plot to show actual vs predicted.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw scatter plot
    ax.scatter(y_h_test, h_predictions, alpha=0.6, s=100, edgecolors='k', linewidths=0.5)

    # Draw diagonal line (ideal prediction line)
    ax.plot([0.8, 2.2], [0.8, 2.2], 'r--', lw=2, label='Perfect prediction')

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_h_test, h_predictions)

    # Set title and labels
    ax.set_title(f'{TARGET_NAMES["H"]} Model: Actual vs Predicted\nAccuracy={accuracy:.4f}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Actual {TARGET_NAMES["H"]}', fontsize=12)
    ax.set_ylabel(f'Predicted {TARGET_NAMES["H"]}', fontsize=12)
    ax.set_xlim(0.8, 2.2)
    ax.set_ylim(0.8, 2.2)
    ax.set_xticks([1, 2])
    ax.set_yticks([1, 2])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_diagonal_H.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"H value diagonal plot saved: {output_dir / 'prediction_diagonal_H.png'}")


def plot_feature_importance(
    output_dir: Path,
    model,
    x_data: pd.DataFrame,
    model_name: str,
    target_name: str,
):
    """
    Plot model feature importance.

    Parameters:
        output_dir: output directory
        model: trained model (Pipeline)
        x_data: input data
        model_name: model name (for filename)
        target_name: target variable identifier (G or H)
    """

    full_target_name = TARGET_NAMES.get(target_name, target_name)
    print(f"\nGenerating Feature Importance plot for {full_target_name} model ({model_name})...")

    try:
        # Get actual model object
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['model']
        else:
            actual_model = model

        feature_names = x_data.columns.tolist()
        importances = None

        # Extract feature importance by model type
        if hasattr(actual_model, 'feature_importances_'):
            # Tree models: XGBoost, RandomForest, GradientBoosting
            importances = actual_model.feature_importances_
            importance_type = 'Feature Importance'
        elif hasattr(actual_model, 'coef_'):
            # Linear models: Lasso
            importances = np.abs(actual_model.coef_)
            importance_type = 'Absolute Coefficient'
        else:
            print(f"{model_name} does not support direct feature importance extraction, skipping...")
            return

        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        # Draw horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(importance_df['Importance'] / importance_df['Importance'].max())
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel(importance_type, fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Importance: {model_name} for {full_target_name}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'])):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                   va='center', ha='left', fontsize=9, fontweight='bold')

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f"feature_importance_{target_name}_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Feature Importance plot saved: {output_path}")

        # Save feature importance data to Excel
        excel_path = output_dir / f"feature_importance_{target_name}_{model_name}.xlsx"
        importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
        importance_df_sorted.to_excel(excel_path, index=False)
        print(f"Feature Importance data saved to Excel: {excel_path}")

    except Exception as e:
        print(f"Error generating Feature Importance plot for {target_name} model ({model_name}): {e}")
        print(f"Skipping Feature Importance plot generation, continuing with other tasks...")


def plot_shap_summary(
    output_dir: Path,
    model,
    x_data: pd.DataFrame,
    model_name: str,
    target_name: str,
):
    """
    Generate SHAP summary plot for given model.

    Parameters:
        output_dir: output directory
        model: trained model (Pipeline)
        x_data: input data
        model_name: model name (for filename)
        target_name: target variable identifier (G or H)
    """

    full_target_name = TARGET_NAMES.get(target_name, target_name)
    print(f"\nGenerating SHAP plot for {full_target_name} model...")

    try:
        # Get processed data (after imputation and standardization)
        # Extract preprocessing steps from Pipeline
        if hasattr(model, 'named_steps'):
            # Apply all preprocessing steps
            x_processed = x_data.copy()
            for step_name in model.named_steps:
                if step_name != 'model':
                    x_processed = model.named_steps[step_name].transform(x_processed)

            # Get actual model
            actual_model = model.named_steps['model']
        else:
            x_processed = x_data.copy()
            actual_model = model

        # Select appropriate SHAP explainer
        # Use TreeExplainer for tree models, KernelExplainer for others
        if hasattr(actual_model, 'get_booster'):  # XGBoost
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(x_processed)
        elif hasattr(actual_model, 'estimators_'):  # RandomForest, GradientBoosting
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(x_processed)
        else:
            # For other models (Lasso, SVR, MLP) use sampling
            # Sample to accelerate computation
            sample_size = min(100, len(x_processed))
            x_sample = x_processed[:sample_size] if isinstance(x_processed, np.ndarray) else x_processed.iloc[:sample_size]

            explainer = shap.KernelExplainer(actual_model.predict, x_sample)
            shap_values = explainer.shap_values(x_processed)

        # Create SHAP summary plot
        plt.figure(figsize=(10, 6))

        # If x_processed is numpy array, convert back to DataFrame to preserve column names
        if isinstance(x_processed, np.ndarray):
            x_processed_df = pd.DataFrame(x_processed, columns=x_data.columns)
        else:
            x_processed_df = x_processed

        shap.summary_plot(
            shap_values,
            x_processed_df,
            show=False,
            plot_size=(10, 6)
        )

        plt.title(f'SHAP Summary: {model_name} for {full_target_name}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        output_path = output_dir / f"shap_values_{target_name}_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP plot saved: {output_path}")

        # Export SHAP values to Excel
        shap_df = pd.DataFrame(shap_values, columns=x_data.columns)
        shap_df.insert(0, 'Sample_Index', range(len(shap_df)))

        # Add original feature values
        for col in x_data.columns:
            shap_df[f'{col}_value'] = x_data[col].values

        excel_path = output_dir / f"shap_values_{target_name}_{model_name}.xlsx"
        shap_df.to_excel(excel_path, index=False)
        print(f"SHAP values saved to Excel: {excel_path}")

    except Exception as e:
        print(f"Error generating SHAP plot for {target_name} model: {e}")
        print(f"Skipping SHAP plot generation, continuing with other tasks...")


def plot_dimensionality_reduction(
    output_dir: Path,
    x_train: pd.DataFrame,
    y_g_train: pd.Series,
    y_h_train: pd.Series,
    optimized_recipes: list[tuple[np.ndarray, float, int]],
):
    """
    Use t-SNE for dimensionality reduction visualization, show training set and optimized recipe points.

    Only plot tSNE1-tSNE2 colored by G value.

    Parameters:
        output_dir: Output directory
        x_train: Training set input features
        y_g_train: Training set G values
        y_h_train: Training set H values
        optimized_recipes: 3 optimized recipes
    """

    print("\nGenerating dimensionality reduction plots...")

    # Prepare predicted points data
    predicted_points = np.array([recipe for recipe, _, _ in optimized_recipes])
    predicted_g = np.array([pred_g for _, pred_g, _ in optimized_recipes])
    predicted_h = np.array([pred_h for _, _, pred_h in optimized_recipes])

    # Merge training set and predicted points for dimensionality reduction
    x_train_array = x_train.values
    all_data = np.vstack([x_train_array, predicted_points])

    # === t-SNE dimensionality reduction (3 components) ===
    tsne = TSNE(n_components=3, random_state=RANDOM_STATE, perplexity=min(30, len(all_data)-1))
    tsne_result = tsne.fit_transform(all_data)

    # Separate training set and predicted points
    tsne_train = tsne_result[:len(x_train_array)]
    tsne_predicted = tsne_result[len(x_train_array):]

    # Plot only tSNE1-tSNE2 colored by G value
    idx_x, idx_y = 0, 1
    label_x, label_y = 'tSNE1', 'tSNE2'
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by G value
    scatter = ax.scatter(
        tsne_train[:, idx_x], tsne_train[:, idx_y],
        c=y_g_train, cmap='viridis', alpha=0.6, s=50,
        edgecolors='k', linewidths=0.5, label='Training Set'
    )
    ax.scatter(
        tsne_predicted[:, idx_x], tsne_predicted[:, idx_y],
        c='red', marker='*', s=200, edgecolors='none', linewidths=0,
        label='Optimized Recipes', zorder=5
    )

    for i, (x, y) in enumerate(zip(tsne_predicted[:, idx_x], tsne_predicted[:, idx_y]), 1):
        ax.annotate(f'{i}', (x, y), fontsize=10, fontweight='bold',
                    ha='center', va='center', color='white')

    ax.set_xlabel(f'{label_x}', fontsize=12)
    ax.set_ylabel(f'{label_y}', fontsize=12)
    ax.set_title(f't-SNE {label_x}-{label_y}: Training Set (colored by {TARGET_NAMES["G"]}) + Optimized Recipes',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(TARGET_NAMES['G'], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f"dimensionality_reduction_tSNE_{label_x}_{label_y}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"t-SNE {label_x}-{label_y} plot saved: {output_dir / f'dimensionality_reduction_tSNE_{label_x}_{label_y}.png'}")

    # === Save dimensionality reduction coordinates to Excel ===
    # Only save tSNE1, tSNE2, G columns
    tsne_train_df = pd.DataFrame({
        'tSNE1': tsne_train[:, 0],
        'tSNE2': tsne_train[:, 1],
        'G': y_g_train.values
    })
    tsne_train_df['Data_Type'] = 'Training Set'

    tsne_predicted_df = pd.DataFrame({
        'tSNE1': tsne_predicted[:, 0],
        'tSNE2': tsne_predicted[:, 1],
        'G': predicted_g
    })
    tsne_predicted_df['Data_Type'] = 'Optimized Recipe'
    tsne_predicted_df['Recipe_ID'] = range(1, len(predicted_points) + 1)

    excel_path = output_dir / "dimensionality_reduction_coordinates.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        tsne_all_df = pd.concat([tsne_train_df, tsne_predicted_df], ignore_index=True)
        # Only keep tSNE1, tSNE2, G columns (plus Data_Type for reference)
        tsne_all_df = tsne_all_df[['tSNE1', 'tSNE2', 'G', 'Data_Type']]
        tsne_all_df.to_excel(writer, sheet_name='t-SNE', index=False)

    print(f"Dimensionality reduction coordinates saved to Excel: {excel_path}")


def save_datasets_to_excel(
    output_dir: Path,
    x_train: pd.DataFrame,
    y_g_train: pd.Series,
    y_h_train: pd.Series,
    x_test: pd.DataFrame,
    y_g_test: pd.Series,
    y_h_test: pd.Series,
):
    """
    Save training and test sets to Excel file.
    """

    print("\nSaving training and test sets to Excel...")

    # Merge training set data
    train_data = x_train.copy()
    train_data['G'] = y_g_train.values
    train_data['H'] = y_h_train.values

    # Merge test set data
    test_data = x_test.copy()
    test_data['G'] = y_g_test.values
    test_data['H'] = y_h_test.values

    # Save to Excel, one sheet per dataset
    excel_path = output_dir / "train_test_datasets.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        train_data.to_excel(writer, sheet_name='Training Set', index=False)
        test_data.to_excel(writer, sheet_name='Test Set', index=False)

    print(f"Training and test sets saved to: {excel_path}")
    print(f"  - Training Set: {len(train_data)} samples")
    print(f"  - Test Set: {len(test_data)} samples")


def save_results(
    output_dir: Path,
    x_train: pd.DataFrame,
    y_g_train: pd.Series,
    y_h_train: pd.Series,
    x_test: pd.DataFrame,
    y_g_test: pd.Series,
    y_h_test: pd.Series,
    comparison_g: pd.DataFrame,
    best_g_model_name: str,
    best_g_model,
    best_estimators_g: dict,
    best_h_model,
    test_predictions_g: dict,
    h_predictions: np.ndarray,
    optimized_recipes: list[tuple[np.ndarray, float, int]],
):
    """
    Save models, comparison results, recipes and predictions.

    All CSVs use utf-8-sig encoding
    so Chinese characters display correctly when opened in Excel.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training and test sets to Excel
    save_datasets_to_excel(
        output_dir, x_train, y_g_train, y_h_train,
        x_test, y_g_test, y_h_test
    )

    # Plot G value diagonal
    plot_prediction_diagonal(output_dir, y_g_test, test_predictions_g, comparison_g)

    # Generate Feature Importance plot: only G_GBR model
    if "GBR" in best_estimators_g:
        plot_feature_importance(
            output_dir,
            best_estimators_g["GBR"],
            x_train,
            "GBR",
            "G"
        )

    # Generate SHAP plot: best G model
    plot_shap_summary(
        output_dir,
        best_g_model,
        x_train,
        best_g_model_name,
        "G"
    )

    # Generate dimensionality reduction plot
    plot_dimensionality_reduction(
        output_dir,
        x_train,
        y_g_train,
        y_h_train,
        optimized_recipes
    )

    # G model test metrics and best parameters
    comparison_g_output = comparison_g[['Model', 'Test_RMSE', 'Test_R2']]
    comparison_g_output.to_csv(
        output_dir / "model_comparison_G.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save 3 optimized recipes
    recipe_data_list = []
    for idx, (recipe, predicted_g, predicted_h) in enumerate(optimized_recipes, 1):
        recipe_data = {
            "Recipe_ID": idx,
        }
        recipe_data.update({
            column: float(value)
            for column, value in zip(x_train.columns, recipe)
        })
        recipe_data["Predicted_G"] = float(predicted_g)
        recipe_data["Predicted_H"] = int(predicted_h)
        recipe_data["Objective"] = "G_max_H_1"
        recipe_data_list.append(recipe_data)

    pd.DataFrame(recipe_data_list).to_csv(
        output_dir / "optimal_recipes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # G test set observed values and all model predictions
    prediction_data_g = pd.DataFrame(
        {
            "Observed_G": y_g_test.to_numpy(),
        }
    )

    for model_name, predictions in test_predictions_g.items():
        prediction_data_g[
            f"{model_name}_Test_Predicted_G"
        ] = predictions

    prediction_data_g.to_csv(
        output_dir / "test_predictions_G.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    """Execute in order: read data, train and compare, optimize recipes, save results."""

    # Suppress sklearn convergence warnings on small samples, keep output clean
    warnings.filterwarnings("ignore")

    # Use prediction.py directory as project directory,
    # so not affected by current terminal working directory
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / OUTPUT_DIR_NAME

    print(f"Project directory: {project_dir}")

    csv_path = find_csv_file(project_dir)
    print(f"Reading CSV: {csv_path}")

    x, y_g, y_h = load_training_data(csv_path)

    (
        comparison_g,
        best_g_model_name,
        best_g_model,
        best_estimators_g,
        test_predictions_g,
        best_h_model,
        h_predictions,
        x_train,
        x_test,
        y_g_train,
        y_g_test,
        y_h_train,
        y_h_test,
    ) = train_and_compare_models(x, y_g, y_h)

    print("\n" + "=" * 60)
    print(f"{TARGET_NAMES['G']} model comparison results (sorted by test RMSE, low to high)")
    print("=" * 60)
    print(comparison_g.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"Best {TARGET_NAMES['G']} model: {best_g_model_name}")
    print("=" * 60)

    # Recipe optimization: search for recipe maximizing Single-crystal Ratio of Substrate with hBN Domain Alignment=1
    print("\n" + "=" * 60)
    print(f"Search for optimal recipes (goal: maximize {TARGET_NAMES['G']}, {TARGET_NAMES['H']}=1)")
    print("=" * 60)

    optimized_recipes = []
    # Run optimization multiple times to get 3 different recipes
    for i in range(3):
        print(f"\nSearching for recipe {i+1}...")
        recipe, predicted_g, predicted_h = optimize_recipe(
            model_g=best_g_model,
            model_h=best_h_model,
            x=x_train,
            seed_offset=i * 100  # Use different random seeds
        )
        optimized_recipes.append((recipe, predicted_g, predicted_h))

        print(f"Recipe {i+1}:")
        for column, value in zip(x_train.columns, recipe):
            print(f"  {column} = {value:.10g}")
        print(f"  Predicted {TARGET_NAMES['G']} = {predicted_g:.10g}")
        print(f"  Predicted {TARGET_NAMES['H']} = {predicted_h}")

    save_results(
        output_dir=output_dir,
        x_train=x_train,
        y_g_train=y_g_train,
        y_h_train=y_h_train,
        x_test=x_test,
        y_g_test=y_g_test,
        y_h_test=y_h_test,
        comparison_g=comparison_g,
        best_g_model_name=best_g_model_name,
        best_g_model=best_g_model,
        best_estimators_g=best_estimators_g,
        best_h_model=best_h_model,
        test_predictions_g=test_predictions_g,
        h_predictions=h_predictions,
        optimized_recipes=optimized_recipes,
    )

    print(f"\nResults saved to: {output_dir}")
    print("Output files:")
    print("  train_test_datasets.xlsx (training and test sets)")
    print("  optimal_recipes.csv (3 optimal recipes)")
    print("  model_comparison_G.csv (model comparison for Single-crystal Ratio of Substrate)")
    print("  test_predictions_G.csv (test set predictions for Single-crystal Ratio of Substrate)")
    print("  test_predictions_G.png (test set predictions plot for Single-crystal Ratio of Substrate)")
    print("  feature_importance_G_GBR.png (GBR model feature importance for Single-crystal Ratio of Substrate)")
    print("  feature_importance_G_GBR.xlsx (GBR model feature importance data for Single-crystal Ratio of Substrate)")
    print("  dimensionality_reduction_tSNE_tSNE1_tSNE2.png")
    print("  dimensionality_reduction_coordinates.xlsx (contains tSNE1, tSNE2, Single-crystal Ratio of Substrate columns)")
    print("  shap_values_G_<model_name>.png (SHAP values for Single-crystal Ratio of Substrate)")
    print("  shap_values_G_<model_name>.xlsx (SHAP values data for Single-crystal Ratio of Substrate)")

    print(
        "\nNote: Optimized recipes are model predictions, "
        "must be validated through actual experiments."
    )


if __name__ == "__main__":
    main()
