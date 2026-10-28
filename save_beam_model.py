
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

X = df[feature_columns]
y = df["beam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(
    X_train,
    y_train,
)

joblib.dump(
    model,
    MODEL_PATH,
)

print(f"Model saved to: {MODEL_PATH}")
