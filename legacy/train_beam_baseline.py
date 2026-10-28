from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Legacy scripts live in legacy/; data and model paths are at the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "public_data"
    / "scenario23_dev"
    / "scenario23_beam_dataset.csv"
)

df = pd.read_csv(DATASET_PATH)

print(f"Total samples: {len(df)}")

feature_columns = [f"power_{i}" for i in range(1, 65)]

X = df[feature_columns]
y = df["beam_label"]

print(f"Features: {X.shape}")
print(f"Beam classes: {y.nunique()}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train,
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Accuracy (%): {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)