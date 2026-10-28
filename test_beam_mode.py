from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

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

df = pd.read_csv(DATASET_PATH)

feature_columns = [
    f"power_{i}"
    for i in range(1, 65)
]

model = joblib.load(MODEL_PATH)

sample = df.iloc[[0]]

X_sample = sample[
    feature_columns
]

predicted_beam = model.predict(
    X_sample
)[0]

actual_beam = sample[
    "beam_label"
].iloc[0]

print(f"Actual beam: {actual_beam}")
print(f"Predicted beam: {predicted_beam}")
print(f"Correct: {predicted_beam == actual_beam}")