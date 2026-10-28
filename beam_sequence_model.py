"""GRU sequence model for proactive beam prediction on Scenario 23.

Same task, split and metrics as beam_prediction.py, so the numbers are
directly comparable:

- input: the last WINDOW sweeps (dB relative to peak), sampling gaps, and
  UAV kinematics when the full download is present;
- target: the beam at t + horizon, scored with GroupKFold over seq_index.

Two training targets are compared:

- "hard": cross-entropy on the argmax beam (the usual setup);
- "soft": cross-entropy against softmax(measured dB / TEMPERATURE) of the
  sweep at t + horizon. In 75% of sweeps the best two beams are within
  0.25 dB, so the hard label is largely measurement noise; the soft target
  teaches the network which beams are good for the link.

The network's logits are added to a learned multiple of the last sweep,
so it only has to learn the change (a residual over persistence).

Run:  python beam_sequence_model.py
Writes results/beam_sequence_metrics.json and .csv
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from beam_prediction import BEAMS, FOLDS, RESULTS_DIR, SEED, link_metrics, load_scenario23

WINDOW = 8
HORIZONS = (1, 3)
TEMPERATURE_DB = 0.5
EPOCHS = 40
PATIENCE = 6
BATCH = 128

torch.set_num_threads(max(1, torch.get_num_threads()))


class BeamGRU(nn.Module):
    """GRU over the sweep history with a residual on the last sweep."""

    def __init__(self, input_size: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, BEAMS))
        self.prior_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, sequence: torch.Tensor, last_sweep_db: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(sequence)
        return self.head(output[:, -1]) + self.prior_weight * last_sweep_db


def build_windows(
    power: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
    time_s: np.ndarray,
    kinematics: np.ndarray | None,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Return (window, horizon) samples that stay inside one sequence."""

    relative_db = 10.0 * np.log10(power / power.max(axis=1, keepdims=True))
    gaps = np.diff(time_s, prepend=time_s[0])[:, None]
    per_step = [relative_db, gaps] + ([kinematics] if kinematics is not None else [])
    steps = np.hstack(per_step).astype(np.float32)

    ends = np.array([
        t for t in range(WINDOW - 1, len(labels) - horizon)
        if sequences[t - WINDOW + 1] == sequences[t + horizon]
    ])
    windows = np.stack([steps[t - WINDOW + 1:t + 1] for t in ends])

    return {
        "windows": windows,
        "last_sweep_db": relative_db[ends].astype(np.float32),
        "target": labels[ends + horizon],
        "target_db": relative_db[ends + horizon].astype(np.float32),
        "target_power": power[ends + horizon],
        "groups": sequences[ends],
    }


def soft_targets(
    target_db: np.ndarray,
) -> np.ndarray:
    """Distribution over beams favouring those within a fraction of a dB of the best."""

    scaled = target_db / TEMPERATURE_DB
    scaled -= scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)

    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def train_model(
    data: dict[str, np.ndarray],
    train: np.ndarray,
    validation: np.ndarray,
    soft: bool,
) -> tuple[BeamGRU, np.ndarray, np.ndarray]:
    """Train with early stopping on held-out training sequences."""

    torch.manual_seed(SEED)

    mean = data["windows"][train].reshape(-1, data["windows"].shape[2]).mean(axis=0)
    std = data["windows"][train].reshape(-1, data["windows"].shape[2]).std(axis=0) + 1e-6

    def tensors(index: np.ndarray):
        target = soft_targets(data["target_db"][index]) if soft else np.eye(BEAMS, dtype=np.float32)[data["target"][index]]
        return (
            torch.from_numpy((data["windows"][index] - mean) / std),
            torch.from_numpy(data["last_sweep_db"][index]),
            torch.from_numpy(target),
        )

    train_x, train_last, train_y = tensors(train)
    val_x, val_last, val_y = tensors(validation)

    model = BeamGRU(data["windows"].shape[2])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loss_fn(logits, target):
        return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

    best_loss, best_state, stale = np.inf, None, 0
    generator = torch.Generator().manual_seed(SEED)

    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            optimiser.zero_grad()
            loss_fn(model(train_x[batch], train_last[batch]), train_y[batch]).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x, val_last), val_y).item()

        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, mean, std


def evaluate_horizon(
    data: dict[str, np.ndarray],
) -> list[dict]:
    """GroupKFold over sequences; one inner group split for early stopping."""

    rows = []
    outer = GroupKFold(n_splits=FOLDS)

    for fold, (train_all, test) in enumerate(outer.split(data["windows"], data["target"], data["groups"])):
        inner_train, inner_val = next(GroupKFold(n_splits=5).split(
            train_all, groups=data["groups"][train_all]
        ))
        train, validation = train_all[inner_train], train_all[inner_val]

        target, target_power = data["target"][test], data["target_power"][test]
        rows.append({"model": "persistence", "fold": fold,
                     **link_metrics(data["last_sweep_db"][test], target, target_power),
                     "inference_ms": 0.0})

        for soft in (False, True):
            started = time.perf_counter()
            model, mean, std = train_model(data, train, validation, soft)
            train_s = time.perf_counter() - started

            model.eval()
            started = time.perf_counter()
            with torch.no_grad():
                scores = model(
                    torch.from_numpy((data["windows"][test] - mean) / std),
                    torch.from_numpy(data["last_sweep_db"][test]),
                ).numpy()
            inference_ms = (time.perf_counter() - started) * 1e3 / len(test)

            name = "gru_soft_target" if soft else "gru_hard_label"
            rows.append({"model": name, "fold": fold,
                         **link_metrics(scores, target, target_power),
                         "inference_ms": inference_ms, "train_s": train_s})
            print(f"  fold {fold} {name}: top1 {rows[-1]['top1_acc']:.3f} "
                  f"top3 {rows[-1]['top3_acc']:.3f} power {rows[-1]['power_ratio']:.3f}")

    return rows


def main() -> None:
    """Run both horizons and write the summary."""

    power, labels, sequences, time_s, kinematics = load_scenario23()
    print(f"UAV kinematics: {'yes' if kinematics is not None else 'no'}")

    all_rows = []
    for horizon in HORIZONS:
        data = build_windows(power, labels, sequences, time_s, kinematics, horizon)
        print(f"Horizon t+{horizon}: {len(data['target'])} windows")
        all_rows += [{"horizon": horizon, **row} for row in evaluate_horizon(data)]

    table = pd.DataFrame(all_rows)
    summary = table.drop(columns="fold").groupby(["horizon", "model"]).mean()

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "beam_sequence_folds.csv", index=False)
    summary.round(4).to_csv(RESULTS_DIR / "beam_sequence_metrics.csv")
    (RESULTS_DIR / "beam_sequence_metrics.json").write_text(json.dumps({
        "split": f"GroupKFold({FOLDS}) over seq_index",
        "window": WINDOW,
        "soft_target_temperature_db": TEMPERATURE_DB,
        "uses_uav_kinematics": kinematics is not None,
        "summary": [
            {"horizon": h, "model": m, **values}
            for (h, m), values in summary.round(4).to_dict(orient="index").items()
        ],
    }, indent=2))

    columns = ["top1_acc", "within_noise_acc", "balanced_within_noise_acc", "top3_acc",
               "within_1_beam_acc", "power_ratio", "macro_f1"]
    print()
    print(summary[columns].round(3).to_string())


if __name__ == "__main__":
    main()
