import pandas as pd
import joblib
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from preprocess import preprocess, TARGET_COLUMN

def main():
    if len(sys.argv) < 2:
        
        sys.exit(1)

    test_file = sys.argv[1]
    
    
    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        
        sys.exit(1)

    try:
        pipeline = joblib.load('saved_models/best_model.pkl')
        label_encoder = joblib.load('saved_models/label_encoder.pkl')
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    # =====================================
    # PREPROCESSING (shared with train.py)
    # =====================================

    
    preprocess(df)

    # =====================================
    # TARGET COLUMN (GamePopularity)
    # =====================================

    has_target = TARGET_COLUMN in df.columns

    if has_target:
        # Drop rows where target is missing — do NOT fabricate labels
        missing_target = df[TARGET_COLUMN].isna()
        if missing_target.any():
            df = df.dropna(subset=[TARGET_COLUMN])

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        y_encoded = label_encoder.transform(y)
    else:
        X = df
        y_encoded = None

    # The pipeline handles SimpleImputer (missing values in features), StandardScaler, and OneHotEncoder
    predictions = pipeline.predict(X)
    
    # Decode predictions back to original labels (Low, Medium, High)
    pred_labels = label_encoder.inverse_transform(predictions)
    
    # Save predictions
    output_df = pd.DataFrame({'Predicted_GamePopularity': pred_labels})
    output_df.to_csv('predictions.csv', index=False)
    print("Predictions successfully saved to 'predictions.csv'.")

    # Show Accuracy if target was present
    if has_target:
        accuracy = accuracy_score(y_encoded, predictions)
        bal_accuracy = balanced_accuracy_score(y_encoded, predictions)
        print("\n=====================================")
        print(f"Test Set Accuracy:          {accuracy:.4f}")
        print(f"Test Set Balanced Accuracy: {bal_accuracy:.4f}")
        print("=====================================")
        print("\nClassification Report:")
        print(classification_report(
            y_encoded,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0
        ))

if __name__ == "__main__":
    main()
