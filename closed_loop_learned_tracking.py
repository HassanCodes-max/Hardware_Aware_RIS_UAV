"""Learned beam tracking trained on measured-only observations (closed loop).

The open-loop GRU (beam_sequence_model.py) needs full 64-beam sweeps of
the last 8 steps, which a reduced-training system never has (AUDIT.md H6).
Here every model sees only what a closed-loop tracker measured:

1. Each flight is replayed with the tie-graph tracker (window +/-1, 4 tie
   partners, full sweep every 8 steps, loss fallback; the best policy of
   closed_loop_beam_tracking.py). The log is, per step, the measured
   beams' power in dB relative to the strongest measured beam,
   unmeasured beams filled with FILL_DB, the measured mask, a full-sweep
   flag and the time gap.
2. A model reads the last WINDOW logged steps and predicts the best beam
   at the next step. The training target is the next step's recorded
   full sweep (soft: softmax(dB / 0.5 dB); hard: argmax), which is
   available offline in the dataset, not to the tracker.
3. Deployment, closed loop on held-out flights: at each step the tracker
   measures its usual set plus the model's top-K_EXTRA beams that are not
   already in it. The model's next input is built from what was measured.

Fair baseline: the same tracker given K_EXTRA more tie partners instead of
model beams, at the same budget. If the model doesn't beat that, the
learning adds nothing.

Models: GRU, LSTM, temporal CNN, Transformer encoder, MLP; GRU variants
over hidden size, learning rate, window and target. The headline
configuration is chosen by validation top-4 coverage on flights held out
*inside* the training folds; every configuration's test result is
reported too.

Caveat: training logs come from the tracker alone; in deployment the
model's extra beams add observations, a mild train/deploy shift.

Run:  python closed_loop_learned_tracking.py            (all configurations)
      python closed_loop_learned_tracking.py --smoke    (1 config, 1 fold, 2 epochs)
Writes results/closed_loop_learned_tracking.{json,csv}
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from beam_prediction import BEAMS, FOLDS, NOISE_TOLERANCE_DB, RESULTS_DIR, SEED, load_scenario23
from beam_training_overhead import BEAM_MEASUREMENT_MS, REPORT_ROUND_TRIP_MS, tie_graph
from closed_loop_beam_tracking import DROP_DB, Policy, run_flight

FILL_DB = -20.0
TEMPERATURE_DB = 0.5
BEHAVIOUR = Policy("window +/-1 + 4 tie partners, full every 8", 8, window=1, partners=4)
K_EXTRA = (2, 4)
STEP_FEATURES = 2 * BEAMS + 2  # dB, mask, full-sweep flag, time gap

torch.set_num_threads(max(1, torch.get_num_threads()))


@dataclass(frozen=True)
class Config:
    name: str
    model: str
    hidden: int = 128
    lr: float = 1e-3
    window: int = 8
    soft: bool = True
    epochs: int = 30
    patience: int = 5
    batch: int = 128


CONFIGS = [
    Config("GRU h128", "gru"),
    Config("LSTM h128", "lstm"),
    Config("TCN h128", "tcn"),
    Config("Transformer h64", "transformer", hidden=64),
    Config("MLP h256", "mlp", hidden=256),
    Config("GRU h64", "gru", hidden=64),
    Config("GRU h128 lr3e-4", "gru", lr=3e-4),
    Config("GRU h128 window 4", "gru", window=4),
    Config("GRU h128 window 16", "gru", window=16),
    Config("GRU h128 hard target", "gru", soft=False),
]


# ---------------------------------------------------------------- models

class Recurrent(nn.Module):
    def __init__(self, cell: str, hidden: int):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(STEP_FEATURES, hidden, num_layers=2, batch_first=True, dropout=0.2)

    def forward(self, x):
        return self.rnn(x)[0][:, -1]


class TemporalCNN(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(STEP_FEATURES, hidden, 3, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=2), nn.ReLU(),
        )

    def forward(self, x):
        # Causal: padding is symmetric, so keep the output aligned with the last input step.
        out = self.net(x.transpose(1, 2))
        return out[:, :, x.shape[1] - 1]


class TransformerEncoder(nn.Module):
    def __init__(self, hidden: int, window: int):
        super().__init__()
        self.embed = nn.Linear(STEP_FEATURES, hidden)
        self.position = nn.Parameter(torch.zeros(1, window, hidden))
        layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x):
        return self.encoder(self.embed(x) + self.position[:, :x.shape[1]])[:, -1]


class FlatMLP(nn.Module):
    def __init__(self, hidden: int, window: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(window * STEP_FEATURES, hidden), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class BeamTracker(nn.Module):
    """Sequence body + head, plus a learned multiple of the last observed dB (residual)."""

    def __init__(self, config: Config):
        super().__init__()
        if config.model in ("gru", "lstm"):
            self.body = Recurrent(config.model, config.hidden)
        elif config.model == "tcn":
            self.body = TemporalCNN(config.hidden)
        elif config.model == "transformer":
            self.body = TransformerEncoder(config.hidden, config.window)
        else:
            self.body = FlatMLP(config.hidden, config.window)
        self.head = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.ReLU(),
                                  nn.Linear(config.hidden, BEAMS))
        self.prior_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, last_db):
        return self.head(self.body(x)) + self.prior_weight * last_db


# ---------------------------------------------------------- observations

def step_features(power_row: np.ndarray, mask_row: np.ndarray, full: bool, gap_s: float) -> np.ndarray:
    measured = np.where(mask_row, power_row, np.nan)
    db = 10.0 * np.log10(measured / np.nanmax(measured))
    db = np.where(mask_row, db, FILL_DB)
    return np.concatenate([db, mask_row.astype(float), [float(full), gap_s]]).astype(np.float32)


def flight_log(power: np.ndarray, time_s: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Per-step features of one flight under the behaviour tracker."""

    trace = run_flight(power, BEHAVIOUR, graph)
    gaps = np.diff(time_s, prepend=time_s[0])
    return np.stack([step_features(power[n], trace["mask"][n], trace["full_sweep"][n], gaps[n])
                     for n in range(len(power))])


def windows_of(log: np.ndarray, window: int) -> np.ndarray:
    """Window ending at every step, zero-padded before the flight starts."""

    padded = np.concatenate([np.zeros((window - 1, log.shape[1]), np.float32), log])
    return np.stack([padded[n:n + window] for n in range(len(log))])


def soft_target(power_row: np.ndarray) -> np.ndarray:
    scaled = 10.0 * np.log10(power_row / power_row.max(axis=-1, keepdims=True)) / TEMPERATURE_DB
    weights = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return (weights / weights.sum(axis=-1, keepdims=True)).astype(np.float32)


def build_samples(flights: list[dict], window: int) -> dict[str, np.ndarray]:
    """(window at step n) -> best beam at n + 1, for every step with a successor."""

    x, last, target_soft, target_hard, flight_id = [], [], [], [], []
    for f in flights:
        w = windows_of(f["log"], window)[:-1]
        x.append(w)
        last.append(w[:, -1, :BEAMS])
        target_soft.append(soft_target(f["power"][1:]))
        target_hard.append(f["power"][1:].argmax(axis=1))
        flight_id.append(np.full(len(w), f["sequence"]))
    return {"x": np.concatenate(x), "last": np.concatenate(last), "soft": np.concatenate(target_soft),
            "hard": np.concatenate(target_hard), "flight": np.concatenate(flight_id)}


# -------------------------------------------------------------- training

def train(config: Config, train_set: dict, val_set: dict, seed: int = SEED):
    torch.manual_seed(seed)
    flat = train_set["x"].reshape(-1, STEP_FEATURES)
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-6

    def tensors(data):
        target = data["soft"] if config.soft else np.eye(BEAMS, dtype=np.float32)[data["hard"]]
        return (torch.from_numpy((data["x"] - mean) / std), torch.from_numpy(data["last"]),
                torch.from_numpy(target))

    tx, tl, ty = tensors(train_set)
    vx, vl, vy = tensors(val_set)
    model = BeamTracker(config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    def loss_fn(logits, target):
        return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

    best, best_state, stale = np.inf, None, 0
    generator = torch.Generator().manual_seed(seed)
    for _ in range(config.epochs):
        model.train()
        order = torch.randperm(len(tx), generator=generator)
        for start in range(0, len(order), config.batch):
            batch = order[start:start + config.batch]
            optimiser.zero_grad()
            loss_fn(model(tx[batch], tl[batch]), ty[batch]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(vx, vl), vy).item()
        if val_loss < best - 1e-4:
            best, stale, best_state = val_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(vx, vl).numpy()
    top4 = np.argsort(-logits, axis=1)[:, :4]
    val_top4 = float(np.mean((top4 == val_set["hard"][:, None]).any(axis=1)))

    return model, mean, std, val_top4


# ------------------------------------------------------------ deployment

def deploy_flight(power, time_s, graph, model, mean, std, window, extra) -> dict[str, np.ndarray]:
    """Closed loop: tracker set + model's top-`extra` beams; model input from measured beams only."""

    steps = len(power)
    gaps = np.diff(time_s, prepend=time_s[0])
    history = np.zeros((window, STEP_FEATURES), np.float32)
    chosen = np.empty(steps, dtype=int)
    beams = np.empty(steps, dtype=int)
    latency = np.empty(steps)
    current, reference, suggestion = None, None, np.array([], dtype=int)

    for n in range(steps):
        mask = np.zeros(BEAMS, dtype=bool)
        full = current is None or n % BEHAVIOUR.full_every == 0
        if full:
            mask[:] = True
            latency[n] = BEAMS * BEAM_MEASUREMENT_MS
        else:
            mask[max(0, current - BEHAVIOUR.window):current + BEHAVIOUR.window + 1] = True
            mask[graph[current, :BEHAVIOUR.partners]] = True
            mask[[b for b in suggestion if not mask[b]][:extra]] = True
            latency[n] = mask.sum() * BEAM_MEASUREMENT_MS
            best = int(np.flatnonzero(mask)[power[n, mask].argmax()])
            if 10.0 * np.log10(power[n, best] / reference) < -DROP_DB:
                latency[n] += REPORT_ROUND_TRIP_MS + (BEAMS - mask.sum()) * BEAM_MEASUREMENT_MS
                mask[:] = True
                full = True
        current = int(np.flatnonzero(mask)[power[n, mask].argmax()])
        reference = power[n, current]
        chosen[n], beams[n] = current, mask.sum()

        history = np.roll(history, -1, axis=0)
        history[-1] = step_features(power[n], mask, full, gaps[n])
        with torch.no_grad():
            logits = model(torch.from_numpy(((history - mean) / std)[None]),
                           torch.from_numpy(history[None, -1, :BEAMS]))[0].numpy()
        suggestion = np.argsort(-logits)

    return {"chosen": chosen, "beams": beams, "latency": latency}


def tracker_flight(power, graph, partners) -> dict[str, np.ndarray]:
    """Baseline at the same budget: the behaviour tracker with more tie partners."""

    policy = Policy("baseline", BEHAVIOUR.full_every, window=BEHAVIOUR.window, partners=partners)
    trace = run_flight(power, policy, graph)
    return {"chosen": trace["chosen"], "beams": trace["beams"], "latency": trace["latency"]}


def summarise(name: str, traces: list[tuple[np.ndarray, dict]]) -> dict:
    exact, noise, ratio, beams, latency = [], [], [], [], []
    for power, trace in traces:
        rows = np.arange(len(power))
        used = power[rows, trace["chosen"]]
        exact.append(trace["chosen"] == power.argmax(axis=1))
        noise.append(10.0 * np.log10(power.max(axis=1) / used) <= NOISE_TOLERANCE_DB)
        ratio.append(used / power.max(axis=1))
        beams.append(trace["beams"])
        latency.append(trace["latency"])
    return {"policy": name, "exact_top1": float(np.mean(np.concatenate(exact))),
            "within_noise": float(np.mean(np.concatenate(noise))),
            "power_ratio": float(np.mean(np.concatenate(ratio))),
            "beams_mean": float(np.mean(np.concatenate(beams))),
            "latency_ms_mean": float(np.mean(np.concatenate(latency)))}


def inference_latency_ms(model: nn.Module, window: int, quantise: bool) -> float:
    """Median batch-1 CPU latency, single thread (laptop CPU, not an embedded board)."""

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    if quantise:
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8)
    x, last = torch.randn(1, window, STEP_FEATURES), torch.randn(1, BEAMS)
    with torch.inference_mode():
        for _ in range(30):
            model(x, last)
        times = []
        for _ in range(300):
            start = time.perf_counter()
            model(x, last)
            times.append((time.perf_counter() - start) * 1e3)
    torch.set_num_threads(threads)
    return float(np.median(times))


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configs = [Config("GRU smoke", "gru", epochs=2)] if args.smoke else CONFIGS

    power, _, sequences, time_s, _ = load_scenario23()
    folds = list(GroupKFold(n_splits=FOLDS).split(power, groups=sequences))
    if args.smoke:
        folds = folds[:1]

    def flights_of(index: np.ndarray, graph: np.ndarray) -> list[dict]:
        out = []
        for sequence in np.unique(sequences[index]):
            rows = np.flatnonzero(sequences == sequence)
            rows = rows[np.argsort(time_s[rows], kind="stable")]
            if len(rows) < 2:
                continue
            out.append({"sequence": int(sequence), "power": power[rows], "time": time_s[rows],
                        "log": flight_log(power[rows], time_s[rows], graph)})
        return out

    rows, baselines = [], {}
    for fold, (train_index, test_index) in enumerate(folds):
        graph = tie_graph(power[train_index])
        train_flights = flights_of(train_index, graph)
        test_flights = flights_of(test_index, graph)

        # Validation flights for early stopping and model selection, inside the training folds.
        train_groups = np.array([f["sequence"] for f in train_flights])
        inner_train, inner_val = next(GroupKFold(n_splits=5).split(train_groups, groups=train_groups))

        for partners in (BEHAVIOUR.partners,) + tuple(BEHAVIOUR.partners + k for k in K_EXTRA):
            name = f"tracker, {partners} tie partners"
            baselines.setdefault(name, []).extend(
                (f["power"], tracker_flight(f["power"], graph, partners)) for f in test_flights)

        for config in configs:
            started = time.perf_counter()
            train_set = build_samples([train_flights[i] for i in inner_train], config.window)
            val_set = build_samples([train_flights[i] for i in inner_val], config.window)
            model, mean, std, val_top4 = train(config, train_set, val_set)
            train_s = time.perf_counter() - started
            for extra in K_EXTRA:
                traces = [(f["power"], deploy_flight(f["power"], f["time"], graph, model, mean, std,
                                                     config.window, extra)) for f in test_flights]
                rows.append({"config": config.name, "fold": fold, "extra_beams": extra,
                             "val_top4": val_top4, "train_s": train_s, "traces": traces,
                             **asdict(config)})
            print(f"fold {fold} {config.name}: val top-4 {val_top4:.3f}, {train_s:.0f} s", flush=True)

            if fold == 0:
                rows[-1]["latency_fp32_ms"] = inference_latency_ms(model, config.window, quantise=False)
                rows[-1]["latency_int8_ms"] = inference_latency_ms(model, config.window, quantise=True)

    # Pool held-out traces over folds.
    summary = []
    for name, traces in baselines.items():
        summary.append({"config": name, "extra_beams": 0, **summarise(name, traces)})
    frame = pd.DataFrame(rows)
    for (name, extra), group in frame.groupby(["config", "extra_beams"], sort=False):
        traces = [t for fold_traces in group["traces"] for t in fold_traces]
        first = group.iloc[0]
        latency = group.dropna(subset=["latency_fp32_ms"]) if "latency_fp32_ms" in group else group.iloc[:0]
        summary.append({
            "config": name, "extra_beams": int(extra),
            "val_top4_mean": float(group["val_top4"].mean()),
            "train_s_per_fold": float(group["train_s"].mean()),
            "inference_fp32_ms": float(latency["latency_fp32_ms"].iloc[0]) if len(latency) else None,
            "inference_int8_ms": float(latency["latency_int8_ms"].iloc[0]) if len(latency) else None,
            "model": first["model"], "hidden": int(first["hidden"]), "lr": float(first["lr"]),
            "window": int(first["window"]), "soft": bool(first["soft"]),
            **summarise(f"{name} + {extra} model beams", traces),
        })
    table = pd.DataFrame(summary)

    selected = (table.dropna(subset=["val_top4_mean"]).sort_values("val_top4_mean", ascending=False)
                .iloc[0]["config"]) if table["val_top4_mean"].notna().any() else None

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    table.to_csv(RESULTS_DIR / f"closed_loop_learned_tracking{suffix}.csv", index=False)
    (RESULTS_DIR / f"closed_loop_learned_tracking{suffix}.json").write_text(json.dumps({
        "behaviour_policy": asdict(BEHAVIOUR),
        "selected_by_validation": selected,
        "rows": table.to_dict(orient="records"),
    }, indent=2, default=str))

    pd.set_option("display.width", 250)
    columns = ["config", "extra_beams", "val_top4_mean", "exact_top1", "within_noise", "beams_mean",
               "latency_ms_mean", "inference_fp32_ms", "inference_int8_ms"]
    print(table[[c for c in columns if c in table]].round(4).to_string(index=False))
    print(f"\nSelected by validation top-4: {selected}")


if __name__ == "__main__":
    main()
