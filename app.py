import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    mean_squared_error, r2_score,
)

from preprocess import preprocess, TARGET_COLUMN
from regression_preprocess import regression_preprocess, REGRESSION_TARGET

# =====================================
# LOAD MODELS AT STARTUP
# =====================================

# --- Classification models ---
clf_pipeline = None
label_encoder = None
try:
    clf_pipeline = joblib.load('saved_models/best_model.pkl')
    label_encoder = joblib.load('saved_models/label_encoder.pkl')
except Exception:
    print("WARNING: Classification model not found. Run train.py first.")

# --- Regression models ---
reg_model = None
reg_scaler = None
reg_features = None
try:
    reg_model = joblib.load('saved_models/best_regression_model.pkl')
    reg_scaler = joblib.load('saved_models/regression_scaler.pkl')
    reg_features = joblib.load('saved_models/regression_feature_columns.pkl')
except Exception:
    print("WARNING: Regression model not found. Run regression_train.py first.")


# =====================================
# CLASSIFICATION HANDLER
# =====================================

def classify(file):
    """Upload CSV → Classification predictions + accuracy."""
    if clf_pipeline is None or label_encoder is None:
        return "ERROR: Classification model not loaded. Run train.py first.", pd.DataFrame()

    df = pd.read_csv(file)
    preprocess(df)

    has_target = TARGET_COLUMN in df.columns

    if has_target:
        df = df.dropna(subset=[TARGET_COLUMN])
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        y_encoded = label_encoder.transform(y)
    else:
        X = df
        y_encoded = None

    predictions = clf_pipeline.predict(X)
    pred_labels = label_encoder.inverse_transform(predictions)

    results_df = X.copy()
    results_df['Predicted_GamePopularity'] = pred_labels

    if has_target:
        acc = accuracy_score(y_encoded, predictions)
        bal_acc = balanced_accuracy_score(y_encoded, predictions)
        cm = confusion_matrix(y_encoded, predictions)
        classes = label_encoder.classes_

        txt = (
            f"Accuracy:             {acc:.4f}\n"
            f"Balanced Accuracy:    {bal_acc:.4f}\n\n"
            f"Confusion Matrix ({', '.join(classes)}):\n"
        )
        for row in cm:
            txt += f"  {list(row)}\n"
    else:
        txt = "No 'GamePopularity' column found — accuracy cannot be calculated."

    return txt, results_df


# =====================================
# REGRESSION HANDLER
# =====================================

def regress(file):
    """Upload CSV → Regression predictions + MSE/R²."""
    if reg_model is None or reg_scaler is None or reg_features is None:
        return "ERROR: Regression model not loaded. Run regression_train.py first.", pd.DataFrame()

    df = pd.read_csv(file, low_memory=False)
    
    has_target = REGRESSION_TARGET in df.columns
    if has_target:
        # Force target to numeric BEFORE preprocessing so clipping doesn't fail
        df[REGRESSION_TARGET] = pd.to_numeric(df[REGRESSION_TARGET], errors='coerce')
        df = df.dropna(subset=[REGRESSION_TARGET])

    regression_preprocess(df, is_training=False)

    if has_target and REGRESSION_TARGET in df.columns:
        y = df[REGRESSION_TARGET].copy()
        X = df.drop(columns=[REGRESSION_TARGET])
    else:
        X = df.copy()
        y = None

    # Align features
    for col in reg_features:
        if col not in X.columns:
            X[col] = 0
    X = X[reg_features]
    X = X.fillna(0)

    # Scale
    X_scaled = pd.DataFrame(
        reg_scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )

    predictions = reg_model.predict(X_scaled)

    results_df = X.copy()
    results_df['Predicted_RecommendationCount'] = predictions

    if has_target and y is not None:
        valid = y.notna()
        y_valid = y[valid].values
        pred_valid = predictions[valid.values]

        mse = mean_squared_error(y_valid, pred_valid)
        r2 = r2_score(y_valid, pred_valid)

        txt = (
            f"MSE:  {mse:.4f}\n"
            f"R²:   {r2:.4f}\n"
        )
    else:
        txt = "No 'RecommendationCount' column found — MSE/R² cannot be calculated."

    return txt, results_df


# =====================================
# GRADIO UI — TABBED INTERFACE
# =====================================

with gr.Blocks(title="Steam Game Predictor") as demo:
    gr.Markdown("# 🎮 Steam Game Predictor")
    gr.Markdown("Upload a CSV file to get predictions. Choose the **Classification** or **Regression** tab.")

    with gr.Tabs():
        # --- Classification Tab ---
        with gr.TabItem("Classification (GamePopularity)"):
            gr.Markdown("Predicts **GamePopularity** (Low / Medium / High). Shows Accuracy & Confusion Matrix if the target column exists.")
            clf_file = gr.File(label="Upload CSV", file_types=[".csv"])
            clf_btn = gr.Button("Predict", variant="primary")
            clf_metrics = gr.Textbox(label="Accuracy & Confusion Matrix", lines=8)
            clf_table = gr.Dataframe(label="Predictions")
            clf_btn.click(fn=classify, inputs=clf_file, outputs=[clf_metrics, clf_table])

        # --- Regression Tab ---
        with gr.TabItem("Regression (RecommendationCount)"):
            gr.Markdown("Predicts **RecommendationCount**. Shows MSE & R² if the target column exists.")
            reg_file = gr.File(label="Upload CSV", file_types=[".csv"])
            reg_btn = gr.Button("Predict", variant="primary")
            reg_metrics = gr.Textbox(label="MSE & R² Score", lines=4)
            reg_table = gr.Dataframe(label="Predictions")
            reg_btn.click(fn=regress, inputs=reg_file, outputs=[reg_metrics, reg_table])

if __name__ == "__main__":
    demo.launch()
