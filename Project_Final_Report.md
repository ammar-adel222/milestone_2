# Machine Learning Project Report: Steam Game Prediction Pipeline

## Executive Summary
This project outlines a complete, production-ready machine learning pipeline developed over two distinct milestones. The goal is to accurately predict two targets based on a comprehensive Steam games dataset:
1. **Milestone 1 (Regression):** Predicting the number of recommendations (`RecommendationCount`).
2. **Milestone 2 (Classification):** Predicting the popularity tier (`GamePopularity`).

The architecture strictly separates preprocessing, training, and testing to prevent data leakage and ensures that models can be evaluated on unseen data without retraining, fulfilling all requirements for the final practical exam.

---

## 1. Data Preprocessing Architecture
A critical component of this pipeline is ensuring that preprocessing is applied consistently during both training and inference (testing). To achieve this, two distinct modules were created: `preprocess.py` (Classification) and `regression_preprocess.py` (Regression).

### 1.1 Shared Preprocessing Steps
Both pipelines apply a foundational set of data cleaning steps:
- **Missing Value Imputation:** Missing strings in textual columns are filled with empty strings `""`, and missing values in numerical columns are filled with `0` (or `median` where appropriate). Empty string spaces (`^\s*$`) are coerced to NaN and handled appropriately.
- **Irrelevant Column Removal:** Identifiers (e.g., `QueryID`, `ResponseID`), URLs, Emails, and raw long-text descriptions that do not provide numeric predictive value are dropped.
- **Date Parsing:** `ReleaseDate` is parsed to extract a numerical `ReleaseYear`.
- **Boolean Conversion:** All boolean feature columns (`True`/`False`) are mapped to integer representations (`1`/`0`).

### 1.2 Regression-Specific Preprocessing (Milestone 1)
Predicting `RecommendationCount` requires handling severe right-skewness and extreme outliers.
- **Outlier Clipping:** Numeric columns (including the target `RecommendationCount` during training) are clipped at the **1st and 99th percentiles**. Crucially, the exact percentile boundaries computed during training are saved to `saved_models/regression_quantiles.json`. During test-time inference, these *saved boundaries* are loaded and applied to unseen data, preventing data leakage and guaranteeing consistent model evaluation.
- **Log Transformation:** Highly skewed numerical features (`SteamSpyOwners`, `PriceInitial`, `PriceFinal`) are log-transformed using `log1p` to normalize their distribution.

### 1.3 Handling Missing Features in Unseen Test Sets
As required, the test scripts **never drop rows** containing missing features.
- If a test sample lacks a feature entirely, the pipeline dynamically creates the missing column and fills it with zeroes, ensuring the feature vector aligns perfectly with the model's expected input dimensions.

---

## 2. Model Training and Selection

### 2.1 Milestone 1: Regression (`RecommendationCount`)
Two models were trained and evaluated via `regression_train.py`: **Random Forest Regressor** and **XGBoost Regressor**.

**Training Configuration:**
- Features scaled using `StandardScaler`, saved via `joblib`.
- Target extracted before preprocessing to manage outlier clipping carefully.
- Cross-validation ($CV=3$) applied to validate generalizability.

**Results:**
- **Random Forest:** Achieved the best performance with an $R^2$ score of **0.8308** and a CV $R^2$ of **0.5304**.
- **Model Artifacts:** The best model (`best_regression_model.pkl`), feature column list, and scaler were serialized to the `saved_models/` directory.

### 2.2 Milestone 2: Classification (`GamePopularity`)
The `GamePopularity` target is highly imbalanced. To combat this, an SVM classifier within a robust scikit-learn Pipeline was utilized in `train.py`.

**Training Configuration:**
- **Model:** Support Vector Machine (`SVC`).
- **Imbalance Handling:** `class_weight='balanced'` was utilized to penalize misclassifications of minority classes proportionally.
- **Optimization:** Hyperparameter tuning was conducted using `GridSearchCV` optimized specifically for `balanced_accuracy`.

**Model Artifacts:**
- The finalized pipeline (`best_model.pkl`) and the `LabelEncoder` (`label_encoder.pkl`) were serialized for deployment.

---

## 3. Practical Exam Readiness: Inference Scripts
To evaluate unseen test datasets without retraining, `test.py` and `regression_test.py` were developed.

### 3.1 Workflow of Test Scripts
1. **Load Artifacts:** Loads the pre-trained model, scaler/pipeline, and feature dictionaries via `joblib`.
2. **Process Test Data:** The raw unseen CSV is passed through the respective preprocessing script (`is_training=False`), applying the exact same scaling and outlier quantiles learned during training.
3. **Prediction & Metric Evaluation:**
   - The scripts generate predictions and save them to a new CSV file (`predictions.csv` / `regression_predictions.csv`).
   - If the unseen test file contains the ground truth target column, the script automatically calculates and outputs the required metrics to the terminal.

### 3.2 Required Evaluation Metrics Implemented
- **Regression:** Outputs **Mean Squared Error (MSE)** and **$R^2$ Score**.
- **Classification:** Outputs overall **Accuracy**, **Balanced Accuracy**, and a formatted **Confusion Matrix** to accurately assess performance across the imbalanced `GamePopularity` classes.

---

## 4. Gradio GUI Deployment
To demonstrate the pipeline interactively, a Gradio web application (`app.py`) was deployed.

### Features:
- **Tabbed Interface:** Separate tabs for Classification and Regression tasks.
- **CSV Upload:** Users can upload raw CSV datasets directly into the browser.
- **Automated Validation:** The app applies preprocessing, predicts using the saved offline models, and displays an interactive DataFrame of the results.
- **Live Metrics:** If the uploaded dataset contains the target variable, the UI dynamically renders the MSE/$R^2$ (Regression) or Accuracy/Confusion Matrix (Classification).

## Conclusion
The project successfully meets all requirements for Milestones 1 and 2. The codebase is highly modular, robust against missing test data, effectively manages severe data skewness via persisted quantiles, and successfully evaluates unseen test samples using serialized, pre-trained model artifacts.
