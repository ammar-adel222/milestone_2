import gradio as gr
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from preprocess import preprocess, TARGET_COLUMN

# Load saved models once at startup
pipeline = joblib.load('saved_models/best_model.pkl')
label_encoder = joblib.load('saved_models/label_encoder.pkl')


def predict(file):
    """Read uploaded CSV, preprocess, predict, and return results."""
    df = pd.read_csv(file)
    preprocess(df)

    has_target = TARGET_COLUMN in df.columns

    if has_target:
        # Drop rows with missing target
        df = df.dropna(subset=[TARGET_COLUMN])
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        y_encoded = label_encoder.transform(y)
    else:
        X = df
        y_encoded = None

    predictions = pipeline.predict(X)
    pred_labels = label_encoder.inverse_transform(predictions)

    # Build results dataframe
    results_df = X.copy()
    results_df['Predicted_GamePopularity'] = pred_labels

    # Build accuracy text
    if has_target:
        acc = accuracy_score(y_encoded, predictions)
        bal_acc = balanced_accuracy_score(y_encoded, predictions)
        cm = confusion_matrix(y_encoded, predictions)
        classes = label_encoder.classes_

        accuracy_text = (
            f"Accuracy:             {acc:.4f}\n"
            f"Balanced Accuracy:    {bal_acc:.4f}\n\n"
            f"Confusion Matrix ({', '.join(classes)}):\n"
        )
        for row in cm:
            accuracy_text += f"  {list(row)}\n"
    else:
        accuracy_text = "No 'GamePopularity' column found in the uploaded file — accuracy cannot be calculated."

    return accuracy_text, results_df


demo = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload CSV", file_types=[".csv"]),
    outputs=[
        gr.Textbox(label="Accuracy & Confusion Matrix", lines=8),
        gr.Dataframe(label="Predictions"),
    ],
    title="Game Popularity Predictor",
    description="Upload a CSV file to predict GamePopularity. If the file contains a 'GamePopularity' column, accuracy will be shown.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
