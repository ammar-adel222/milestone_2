"""
Regression Training Script (Milestone 1 → Milestone 2)
------------------------------------------------------
Converts the regression notebook into a proper script that:
  - Trains Random Forest and XGBoost regressors
  - Evaluates with MSE, R², and CV R²
  - Saves the best model, scaler, and feature columns to saved_models/

Usage:
    python regression_train.py                    # defaults to train_data.csv
    python regression_train.py <data_file.csv>    # use a specific file
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from regression_preprocess import regression_preprocess, REGRESSION_TARGET

# =====================================
# LOAD DATASET
# =====================================

file_path = sys.argv[1] if len(sys.argv) > 1 else 'train_data_m1.csv'
print(f'Loading data from: {file_path}')

df = pd.read_csv(file_path)

if REGRESSION_TARGET not in df.columns:
    print(f"\nERROR: Column '{REGRESSION_TARGET}' not found in the dataset.")
    print(f"Available columns: {list(df.columns)}")
    print(f"\nThis script requires data with the '{REGRESSION_TARGET}' column.")
    print("If you have M2 data (GamePopularity), use train.py instead.")
    sys.exit(1)

# =====================================
# PREPROCESSING
# =====================================

regression_preprocess(df, is_training=True)

# =====================================
# TARGET / FEATURES
# =====================================

y = df[REGRESSION_TARGET]
X = df.drop(REGRESSION_TARGET, axis=1)

# =====================================
# SCALING
# =====================================

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Save artifacts
os.makedirs('saved_models', exist_ok=True)
joblib.dump(scaler, 'saved_models/regression_scaler.pkl')
joblib.dump(list(X.columns), 'saved_models/regression_feature_columns.pkl')

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =====================================
# MODELS (from notebook)
# =====================================

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    'XGBoost': XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}

mse_scores = []
r2_scores_list = []
cv_r2_scores = []
training_times = []

best_model = None
best_cv_r2 = -np.inf
best_model_name = ''

# =====================================
# TRAIN AND EVALUATE
# =====================================

for name, model in models.items():

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    cv = cross_val_score(model, X_scaled, y, cv=3, scoring='r2', n_jobs=-1)
    cv_r2 = cv.mean()

    print(f'\n{name}')
    print(f'  MSE:    {mse:.4f}')
    print(f'  R2:     {r2:.4f}')
    print(f'  CV R2:  {cv_r2:.4f}')

    mse_scores.append(mse)
    r2_scores_list.append(r2)
    cv_r2_scores.append(cv_r2)
    training_times.append(train_time)

    if cv_r2 > best_cv_r2:
        best_cv_r2 = cv_r2
        best_model = model
        best_model_name = name

# =====================================
# SAVE BEST MODEL
# =====================================

joblib.dump(best_model, 'saved_models/best_regression_model.pkl')

print('\n=====================================')
print(f'Best Regression Model: {best_model_name}')
print(f'Best CV R2:            {best_cv_r2:.4f}')
print('=====================================')

# =====================================
# CHARTS
# =====================================

model_names = list(models.keys())

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_scores_list)
plt.title('R² Score Comparison')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.savefig('saved_models/regression_r2_chart.png', dpi=150, bbox_inches='tight')

plt.figure(figsize=(8, 5))
plt.bar(model_names, mse_scores)
plt.title('MSE Comparison')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.savefig('saved_models/regression_mse_chart.png', dpi=150, bbox_inches='tight')

plt.figure(figsize=(8, 5))
plt.bar(model_names, training_times)
plt.title('Training Time Comparison')
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.savefig('saved_models/regression_training_time_chart.png', dpi=150, bbox_inches='tight')

print('\nCharts saved to saved_models/')
print('Done.')
