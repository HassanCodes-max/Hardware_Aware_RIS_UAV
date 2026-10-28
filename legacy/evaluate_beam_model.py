from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Legacy scripts live in legacy/; data and model paths are at the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "public_data"
    / "scenario23_dev"
    / "scenario23_beam_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "scenario23_beam_model.pkl"
)

PREDICTIONS_PATH = (
    Path(__file__).resolve().parent
    / "scenario23_beam_predictions.csv"
)

df = pd.read_csv(DATASET_PATH)

feature_columns = [
    f"power_{i}"
    for i in range(1, 65)
]

X = df[feature_columns]
y = df["beam_label"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

model = joblib.load(
    MODEL_PATH
)

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

predictions = pd.DataFrame(
    {
        "actual_beam": y_test.to_numpy(),
        "predicted_beam": y_pred,
    }
)

predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print(f"Test samples: {len(y_test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Predictions saved to: {PREDICTIONS_PATH}")