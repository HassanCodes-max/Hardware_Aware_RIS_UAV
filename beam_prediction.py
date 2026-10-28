"""Proactive beam prediction on DeepSense 6G Scenario 23.

Why the task changed (see AUDIT.md):

- The ground-truth label is exactly argmax(power) + 1 for every sample, so
  "predict the beam from the current 64-beam sweep" is solved by argmax
  with 100% accuracy and is not a learning problem. The first draft's
  Random Forest (80.90%) was worse than that one-liner.
- A random 80/20 split of consecutive 7 Hz samples puts near-identical
  neighbours in train and test. Every model here is evaluated with
  GroupKFold over the recorded flight sequences (seq_index), so test
  sequences are never seen in training.

The task evaluated here is the one a tracker actually faces: given the
sweeps up to time t, choose the beam for time t + h before sweeping.

Top-1 accuracy is capped by the data: in 75% of samples the best and
second-best beams are within 0.25 dB (they are adjacent beams 90% of the
time), so choosing either one is equally good for the link. The link-level
metrics are therefore the received-power ratio of the chosen beam to the
optimal beam, top-k accuracy, and +/-1 beam accuracy.

Run:  python beam_prediction.py
Writes results/beam_prediction_metrics.json and .csv
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "public_data" / "scenario23_dev"
RESULTS_DIR = PROJECT_ROOT / "results"

BEAMS = 64
HISTORY = 3
HORIZONS = (1, 3)
FOLDS = 5
SEED = 42


def load_scenario23() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return power sweeps (N, 64), 0-based labels, sequence ids and times (s)."""

    beams = pd.read_csv(DATA_DIR / "scenario23_beam_dataset.csv")
    manifest = pd.read_csv(DATA_DIR / "scenario23.csv")

    if not (beams["index"].to_numpy() == manifest["index"].to_numpy()).all():
        raise ValueError("Beam dataset and manifest are not row-aligned.")

    power = beams[[f"power_{i}" for i in range(1, BEAMS + 1)]].to_numpy(float)
    labels = beams["beam_label"].to_numpy(int) - 1

    clock = manifest["time_stamp[UTC]"].str.extract(r"(\d+)-(\d+)-(\d+)-(\d+)").astype(float)
    time_s = clock[0] * 3600 + clock[1] * 60 + clock[2] + clock[3] / 1000.0

    return power, labels, manifest["seq_index"].to_numpy(int), time_s.to_numpy()


def build_samples(
    power: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
    time_s: np.ndarray,
    horizon: int,
) -> dict[str, np.ndarray]:
    """
    Build (features at t, target at t + horizon) pairs inside each sequence.

    Features: the last HISTORY sweeps in dB relative to their own peak,
    the beam index history and its first difference, and the sampling gaps.
    """

    relative_db = 10.0 * np.log10(power / power.max(axis=1, keepdims=True))

    rows = []
    for t in range(HISTORY - 1, len(labels) - horizon):
        target = t + horizon
        if sequences[t - HISTORY + 1] != sequences[target]:
            continue
        rows.append(t)

    rows = np.asarray(rows)
    past = [rows - k for k in range(HISTORY)]

    features = np.hstack(
        [relative_db[p] for p in past]
        + [np.stack([labels[p] for p in past], axis=1)]
        + [np.stack([labels[past[k]] - labels[past[k + 1]] for k in range(HISTORY - 1)], axis=1)]
        + [np.stack([time_s[past[k]] - time_s[past[k + 1]] for k in range(HISTORY - 1)], axis=1)]
        + [(time_s[rows + horizon] - time_s[rows])[:, None]]
    )

    return {
        "features": features,
        "target": labels[rows + horizon],
        "target_power": power[rows + horizon],
        "last_sweep_db": relative_db[rows],
        "groups": sequences[rows],
    }


def link_metrics(
    scores: np.ndarray,
    target: np.ndarray,
    target_power: np.ndarray,
) -> dict[str, float]:
    """Score a (samples, 64) ranking of beams against the measured sweep."""

    ranking = np.argsort(-scores, axis=1)
    chosen = ranking[:, 0]
    index = np.arange(len(target))
    best_power = target_power.max(axis=1)

    def top_k(k: int) -> float:
        return float(np.mean((ranking[:, :k] == target[:, None]).any(axis=1)))

    power_ratio = target_power[index, chosen] / best_power
    top3_power_ratio = target_power[index[:, None], ranking[:, :3]].max(axis=1) / best_power

    return {
        "top1_acc": float(np.mean(chosen == target)),
        "top3_acc": top_k(3),
        "top5_acc": top_k(5),
        "within_1_beam_acc": float(np.mean(np.abs(chosen - target) <= 1)),
        "power_ratio": float(power_ratio.mean()),
        "power_loss_db": float(-10.0 * np.log10(power_ratio.mean())),
        "top3_power_ratio": float(top3_power_ratio.mean()),
        "macro_f1": float(f1_score(target, chosen, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(target, chosen, average="weighted", zero_division=0)),
    }


def probability_scores(
    model,
    features: np.ndarray,
    last_sweep_db: np.ndarray,
) -> np.ndarray:
    """Map class probabilities onto all 64 beams; break ties by the last sweep."""

    scores = np.zeros((len(features), BEAMS))
    scores[:, model.classes_] = model.predict_proba(features)

    return scores + 1e-6 * (last_sweep_db - last_sweep_db.min())


def models() -> dict:
    """Return the learned models compared against persistence."""

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=SEED
        ),
    }


def evaluate_horizon(
    samples: dict[str, np.ndarray],
) -> list[dict]:
    """Cross-validate persistence and the learned models over sequences."""

    rows = []
    splitter = GroupKFold(n_splits=FOLDS)

    for fold, (train, test) in enumerate(
        splitter.split(samples["features"], samples["target"], samples["groups"])
    ):
        target = samples["target"][test]
        target_power = samples["target_power"][test]
        last_sweep = samples["last_sweep_db"][test]

        rows.append({"model": "persistence", "fold": fold,
                     **link_metrics(last_sweep, target, target_power), "inference_ms": 0.0})

        for name, model in models().items():
            model.fit(samples["features"][train], samples["target"][train])

            started = time.perf_counter()
            scores = probability_scores(model, samples["features"][test], last_sweep)
            inference_ms = (time.perf_counter() - started) * 1e3 / len(test)

            rows.append({"model": name, "fold": fold,
                         **link_metrics(scores, target, target_power),
                         "inference_ms": inference_ms})

            # Blend the learned ranking with the last measurement: the model
            # supplies motion trends, the sweep supplies a strong prior.
            blended = scores + 0.02 * last_sweep
            rows.append({"model": f"{name}+sweep_prior", "fold": fold,
                         **link_metrics(blended, target, target_power),
                         "inference_ms": inference_ms})

    return rows


def legacy_checks(
    power: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Reproduce the first draft's setup to document why it was replaced."""

    argmax_accuracy = float(np.mean(power.argmax(axis=1) == labels))

    train_x, test_x, train_y, test_y = train_test_split(
        power, labels, test_size=0.2, random_state=42
    )
    legacy = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    ).fit(train_x, train_y)

    sorted_power = np.sort(power, axis=1)
    gap_db = 10.0 * np.log10(sorted_power[:, -1] / sorted_power[:, -2])
    order = np.argsort(-power, axis=1)

    return {
        "argmax_of_current_sweep_acc": argmax_accuracy,
        "legacy_rf_current_sweep_random_split_acc": float(
            accuracy_score(test_y, legacy.predict(test_x))
        ),
        "best_vs_second_beam_within_0.25db_pct": float(100.0 * np.mean(gap_db < 0.25)),
        "best_vs_second_beam_within_0.5db_pct": float(100.0 * np.mean(gap_db < 0.5)),
        "best_and_second_adjacent_pct": float(
            100.0 * np.mean(np.abs(order[:, 0] - order[:, 1]) == 1)
        ),
    }


def main() -> None:
    """Run every horizon and write the summary tables."""

    power, labels, sequences, time_s = load_scenario23()

    print(f"Samples: {len(labels)}, sequences: {len(np.unique(sequences))}")

    legacy = legacy_checks(power, labels)
    for key, value in legacy.items():
        print(f"  {key}: {value:.4f}")

    all_rows = []
    for horizon in HORIZONS:
        samples = build_samples(power, labels, sequences, time_s, horizon)
        print(f"\nHorizon t+{horizon}: {len(samples['target'])} samples")

        for row in evaluate_horizon(samples):
            all_rows.append({"horizon": horizon, **row})

    table = pd.DataFrame(all_rows)
    summary = table.drop(columns="fold").groupby(["horizon", "model"]).mean()

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "beam_prediction_folds.csv", index=False)
    summary.round(4).to_csv(RESULTS_DIR / "beam_prediction_metrics.csv")
    (RESULTS_DIR / "beam_prediction_metrics.json").write_text(json.dumps({
        "split": f"GroupKFold({FOLDS}) over seq_index",
        "history_sweeps": HISTORY,
        "legacy_checks": legacy,
        "summary": [
            {"horizon": h, "model": m, **values}
            for (h, m), values in summary.round(4).to_dict(orient="index").items()
        ],
    }, indent=2))

    columns = ["top1_acc", "top3_acc", "within_1_beam_acc", "power_ratio", "power_loss_db", "macro_f1"]
    print()
    print(summary[columns].round(3).to_string())


if __name__ == "__main__":
    main()
