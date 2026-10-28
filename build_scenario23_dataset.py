from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

SCENARIO23_ROOT = (
    PROJECT_ROOT
    / "public_data"
    / "scenario23_dev"
)

CSV_PATH = SCENARIO23_ROOT / "scenario23.csv"

OUTPUT_PATH = (
    SCENARIO23_ROOT
    / "scenario23_beam_dataset.csv"
)


# ---------------------------------------------------------
# Load Scenario 23 manifest
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)

print(f"Total samples: {len(df)}")


# ---------------------------------------------------------
# Read 64 power values from each sample
# ---------------------------------------------------------

power_features = []

valid_rows = []


for row_number, row in df.iterrows():

    power_path = (
        SCENARIO23_ROOT
        / str(row["unit1_pwr_60ghz"]).replace("./", "")
    )

    if not power_path.exists():
        print(
            f"Skipping missing file: {power_path}"
        )
        continue

    power_values = np.loadtxt(
        power_path,
        dtype=float,
    )

    if power_values.size != 64:
        print(
            f"Skipping unexpected power file: "
            f"{power_path} "
            f"({power_values.size} values)"
        )
        continue

    power_features.append(
        power_values
    )

    valid_rows.append(
        row_number
    )


# ---------------------------------------------------------
# Keep only valid samples
# ---------------------------------------------------------

df_valid = df.iloc[
    valid_rows
].reset_index(
    drop=True
)

X = np.asarray(
    power_features,
    dtype=np.float32,
)

y = df_valid[
    "unit1_beam_index"
].to_numpy(
    dtype=np.int64
)


# ---------------------------------------------------------
# Create feature columns
# ---------------------------------------------------------

feature_columns = [
    f"power_{i + 1}"
    for i in range(64)
]

power_df = pd.DataFrame(
    X,
    columns=feature_columns,
)


# ---------------------------------------------------------
# Add useful metadata
# ---------------------------------------------------------

output_df = pd.concat(
    [
        df_valid[
            [
                "index",
                "unit1_rgb",
                "unit1_pwr_60ghz",
                "time_stamp[UTC]",
            ]
        ],
        power_df,
        pd.Series(
            y,
            name="beam_label",
        ),
    ],
    axis=1,
)


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

output_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print(
    f"Valid samples: {len(output_df)}"
)

print(
    f"Feature shape: {X.shape}"
)

print(
    f"Number of beam classes: "
    f"{output_df['beam_label'].nunique()}"
)

print(
    f"Beam range: "
    f"{output_df['beam_label'].min()} "
    f"to "
    f"{output_df['beam_label'].max()}"
)

print(
    f"\nSaved dataset to:\n"
    f"{OUTPUT_PATH}"
)