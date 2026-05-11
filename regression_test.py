"""
Regression Test Script
----------------------
Loads the saved regression model, preprocesses an unseen CSV,
outputs predictions, and shows MSE / R² if the target column exists.

IMPORTANT: Missing feature values are imputed (never dropped) so that
every test row receives a prediction.

Usage:
    python regression_test.py <test_csv_path>
"""

import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from regression_preprocess import regression_preprocess, REGRESSION_TARGET


def main():
    if len(sys.argv) < 2:
        print("Usage: python regression_test.py <test_csv_path>")
        sys.exit(1)

    test_file = sys.argv[1]

    # =====================================
    # LOAD DATA
    # =====================================

    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: File '{test_file}' not found.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from '{test_file}'.")

    # =====================================
    # LOAD SAVED ARTIFACTS
    # =====================================

    try:
        model = joblib.load('saved_models/best_regression_model.pkl')
        scaler = joblib.load('saved_models/regression_scaler.pkl')
        feature_columns = joblib.load('saved_models/regression_feature_columns.pkl')
    except Exception as e:
        print(f"Error loading saved models: {e}")
        print("Make sure you've run regression_train.py first.")
        sys.exit(1)

    has_target = REGRESSION_TARGET in df.columns
    if has_target:
        df[REGRESSION_TARGET] = pd.to_numeric(df[REGRESSION_TARGET], errors='coerce')
        df = df.dropna(subset=[REGRESSION_TARGET])

    # =====================================
    # PREPROCESSING (shared, test mode)
    # =====================================

    regression_preprocess(df, is_training=False)

    # =====================================
    # SEPARATE TARGET (if present)
    # =====================================

    if has_target and REGRESSION_TARGET in df.columns:
        y = df[REGRESSION_TARGET].copy()
        X = df.drop(columns=[REGRESSION_TARGET])
    else:
        X = df.copy()
        y = None

    # =====================================
    # ALIGN FEATURES WITH TRAINING SET
    # =====================================

    # Add any missing columns (filled with 0)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Keep only training columns, in training order
    X = X[feature_columns]

    # =====================================
    # HANDLE MISSING VALUES
    # (Cannot drop test rows — fill with 0, scaler handles the rest)
    # =====================================

    X = X.fillna(0)

    # =====================================
    # SCALE (using saved training scaler)
    # =====================================

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )

    # =====================================
    # PREDICT
    # =====================================

    predictions = model.predict(X_scaled)

    # =====================================
    # SAVE PREDICTIONS
    # =====================================

    output_df = pd.DataFrame({'Predicted_RecommendationCount': predictions})
    output_df.to_csv('regression_predictions.csv', index=False)
    print("Predictions saved to 'regression_predictions.csv'.")

    # =====================================
    # METRICS (if target was present)
    # =====================================

    if has_target and y is not None:
        valid = y.notna()
        y_valid = y[valid].values
        pred_valid = predictions[valid.values]

        mse = mean_squared_error(y_valid, pred_valid)
        r2 = r2_score(y_valid, pred_valid)

        print('\n=====================================')
        print(f'  MSE:  {mse:.4f}')
        print(f'  R2:   {r2:.4f}')
        print('=====================================')


if __name__ == "__main__":
    main()
