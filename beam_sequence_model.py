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

A third model, "gru_soft_aug_ens", averages three soft-target GRUs
(different seeds) trained with beam-axis augmentation: each batch is
shifted by up to +/-MAX_SHIFT beams and mirrored with probability 0.5,
applied identically to the history, the last sweep and the target. A UAV
at a slightly different angle sees a shifted beam pattern, so this adds
plausible examples of rare beams; vacated edge beams are filled with the
row's floor value, not wrapped around.

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
MAX_SHIFT = 3
ENSEMBLE_SEEDS = (SEED, SEED + 1, SEED + 2)

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


def shift_beams(
    values: torch.Tensor,
    shift: int,
    mirror: bool,
    fill: torch.Tensor,
) -> torch.Tensor:
    """Shift (and optionally mirror) the last axis of 64 beams, filling vacated beams."""

    if mirror:
        values = values.flip(-1)
    if shift == 0:
        return values

    shifted = torch.roll(values, shift, dims=-1)
    if shift > 0:
        shifted[..., :shift] = fill
    else:
        shifted[..., shift:] = fill

    return shifted


def augment_batch(
    windows: torch.Tensor,
    last: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one random beam shift/mirror to raw dB inputs and the soft target."""

    shift = int(torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (1,), generator=generator))
    mirror = bool(torch.rand(1, generator=generator) < 0.5)

    sweeps = windows[..., :BEAMS]
    sweeps = shift_beams(sweeps, shift, mirror, sweeps.amin(dim=-1, keepdim=True))
    windows = torch.cat([sweeps, windows[..., BEAMS:]], dim=-1)

    last = shift_beams(last, shift, mirror, last.amin(dim=-1, keepdim=True))
    target = shift_beams(target, shift, mirror, torch.zeros_like(target[..., :1]))
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    return windows, last, target


def train_model(
    data: dict[str, np.ndarray],
    train: np.ndarray,
    validation: np.ndarray,
    soft: bool,
    seed: int = SEED,
    augment: bool = False,
) -> tuple[BeamGRU, np.ndarray, np.ndarray]:
    """Train with early stopping on held-out training sequences."""

    torch.manual_seed(seed)

    mean_np = data["windows"][train].reshape(-1, data["windows"].shape[2]).mean(axis=0)
    std_np = data["windows"][train].reshape(-1, data["windows"].shape[2]).std(axis=0) + 1e-6
    mean, std = torch.from_numpy(mean_np), torch.from_numpy(std_np)

    def tensors(index: np.ndarray):
        target = soft_targets(data["target_db"][index]) if soft else np.eye(BEAMS, dtype=np.float32)[data["target"][index]]
        return (
            torch.from_numpy(data["windows"][index]),
            torch.from_numpy(data["last_sweep_db"][index]),
            torch.from_numpy(target),
        )

    train_x, train_last, train_y = tensors(train)
    val_x, val_last, val_y = tensors(validation)
    val_x = (val_x - mean) / std

    model = BeamGRU(data["windows"].shape[2])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loss_fn(logits, target):
        return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

    best_loss, best_state, stale = np.inf, None, 0
    generator = torch.Generator().manual_seed(seed)

    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            windows, last, target = train_x[batch], train_last[batch], train_y[batch]
            if augment:
                windows, last, target = augment_batch(windows, last, target, generator)
            optimiser.zero_grad()
            loss_fn(model((windows - mean) / std, last), target).backward()
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
    return model, mean_np, std_np


def evaluate_horizon(
    data: dict[str, np.ndarray],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """
    GroupKFold over sequences; one inner group split for early stopping.

    Also returns out-of-fold beam scores for every sample, so downstream
    experiments (beam_training_overhead.py) use predictions made by a
    model that never saw that sample's sequence.
    """

    rows = []
    out_of_fold = {
        name: np.full((len(data["target"]), BEAMS), np.nan, dtype=np.float32)
        for name in ("gru_hard_label", "gru_soft_target", "gru_soft_aug_ens")
    }
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
            out_of_fold[name][test] = scores
            rows.append({"model": name, "fold": fold,
                         **link_metrics(scores, target, target_power),
                         "inference_ms": inference_ms, "train_s": train_s})
            print(f"  fold {fold} {name}: top1 {rows[-1]['top1_acc']:.3f} "
                  f"top3 {rows[-1]['top3_acc']:.3f} power {rows[-1]['power_ratio']:.3f}")

        started = time.perf_counter()
        members = []
        for seed in ENSEMBLE_SEEDS:
            model, mean, std = train_model(data, train, validation, soft=True, seed=seed, augment=True)
            model.eval()
            with torch.no_grad():
                members.append(torch.log_softmax(model(
                    torch.from_numpy((data["windows"][test] - mean) / std),
                    torch.from_numpy(data["last_sweep_db"][test]),
                ), dim=1).numpy())
        scores = np.mean(members, axis=0)
        train_s = time.perf_counter() - started

        out_of_fold["gru_soft_aug_ens"][test] = scores
        rows.append({"model": "gru_soft_aug_ens", "fold": fold,
                     **link_metrics(scores, target, target_power), "train_s": train_s})
        print(f"  fold {fold} gru_soft_aug_ens: top1 {rows[-1]['top1_acc']:.3f} "
              f"top3 {rows[-1]['top3_acc']:.3f} power {rows[-1]['power_ratio']:.3f}")

    return rows, out_of_fold


def main() -> None:
    """Run both horizons and write the summary."""

    power, labels, sequences, time_s, kinematics = load_scenario23()
    print(f"UAV kinematics: {'yes' if kinematics is not None else 'no'}")

    RESULTS_DIR.mkdir(exist_ok=True)

    all_rows = []
    for horizon in HORIZONS:
        data = build_windows(power, labels, sequences, time_s, kinematics, horizon)
        print(f"Horizon t+{horizon}: {len(data['target'])} windows")
        rows, out_of_fold = evaluate_horizon(data)
        all_rows += [{"horizon": horizon, **row} for row in rows]
        np.savez_compressed(
            RESULTS_DIR / f"beam_sequence_oof_h{horizon}.npz",
            **out_of_fold,
            target=data["target"],
            target_power=data["target_power"],
            last_sweep_db=data["last_sweep_db"],
            groups=data["groups"],
        )

    table = pd.DataFrame(all_rows)
    summary = table.drop(columns="fold").groupby(["horizon", "model"]).mean()

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
