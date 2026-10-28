"""AI-assisted reduced beam training on DeepSense 6G Scenario 23.

A deployed mmWave link does not choose its beam blind. It predicts a
shortlist from the past, measures only those K beams at the target time,
and keeps the strongest. This script scores that two-stage system against
the full 64-beam ground truth (argmax of the complete sweep at t + h):

- exact top-1: the kept beam is the true optimal beam;
- macro-F1 over the 64 beam classes;
- overhead: beams measured / 64 (an exhaustive sweep is 64).

Why this framing: predicting the exact beam blind is capped by
measurement noise. Even a non-causal oracle that averages the sweeps
before and after the target reaches only 79% exact top-1 (computed below
and stored as `noncausal_oracle_ceiling`). Measuring a few beams resolves
near-ties with the real channel instead of guessing them.

Shortlists:
- last_sweep: the K strongest beams of the last measured sweep;
- window: K beams centred on the last best beam (classic local search);
- gru_soft / gru_hard: out-of-fold GRU rankings (beam_sequence_model.py);
- gru_soft+sweep: z-scored GRU log-probabilities plus the z-scored last
  sweep, equally weighted. The GRU carries motion trends, the last sweep
  carries the current fine structure, and each corrects the other.

Tried and rejected: logit adjustment for the class imbalance (Menon et
al., 2021). The beam distribution differs too much between flights for a
training-fold prior to transfer; it lowered every metric.

Adaptive shortlist: the smallest set of GRU beams whose softmax
probability reaches tau (at most K_MAX), so confident cases measure fewer
beams.

Operating points are chosen with nested cross-validation: for each
held-out fold, K (or tau) is the cheapest setting that meets the target
on the other four folds, then applied to the held-out fold.

Run after beam_sequence_model.py:  python beam_training_overhead.py
Writes results/beam_training_overhead.json and .csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from beam_prediction import BEAMS, FOLDS, NOISE_TOLERANCE_DB, load_scenario23

RESULTS_DIR = Path(__file__).resolve().parent / "results"
HORIZONS = (1, 3)
K_VALUES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
TAU_VALUES = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)
K_MAX = 16
TARGET = 0.85
BOOTSTRAP_RESAMPLES = 500


def noncausal_oracle_ceiling() -> dict[str, float]:
    """Exact top-1 / macro-F1 of argmax(mean of sweeps t-1 and t+1) for label t."""

    power, labels, sequences, _, _ = load_scenario23()
    power_db = 10.0 * np.log10(power)
    middle = np.array([
        i for i in range(1, len(labels) - 1)
        if sequences[i - 1] == sequences[i] == sequences[i + 1]
    ])
    guess = ((power_db[middle - 1] + power_db[middle + 1]) / 2.0).argmax(axis=1)

    return {
        "exact_top1": float(np.mean(guess == labels[middle])),
        "macro_f1": float(f1_score(labels[middle], guess, average="macro")),
    }


def shortlist_rankings(
    data: dict,
) -> dict[str, np.ndarray]:
    """Return each shortlist method's beam ranking, shape (samples, 64)."""

    last = data["last_sweep_db"]
    last_best = last.argmax(axis=1)

    # Window: beams ordered by distance from the last best beam
    # (ties broken towards the stronger side of the last sweep).
    distance = np.abs(np.arange(BEAMS)[None, :] - last_best[:, None]).astype(float)
    window_rank = np.argsort(distance - 1e-3 * (last - last.min()), axis=1)

    fused = zscore(np.log(softmax(data["gru_soft_target"]) + 1e-12)) + zscore(last)

    return {
        "last_sweep": np.argsort(-last, axis=1),
        "window": window_rank,
        "gru_hard": np.argsort(-data["gru_hard_label"], axis=1),
        "gru_soft": np.argsort(-data["gru_soft_target"], axis=1),
        "gru_soft+sweep": np.argsort(-fused, axis=1),
    }


def zscore(scores: np.ndarray) -> np.ndarray:
    """Standardise each row so two score sources can be added fairly."""

    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""

    shifted = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(shifted)

    return weights / weights.sum(axis=1, keepdims=True)


def measure_and_pick(
    ranking: np.ndarray,
    shortlist_sizes: np.ndarray,
    target_power: np.ndarray,
) -> np.ndarray:
    """Measure each sample's first K ranked beams and keep the strongest."""

    rows = np.arange(len(ranking))
    measured = np.where(
        np.arange(ranking.shape[1])[None, :] < shortlist_sizes[:, None],
        target_power[rows[:, None], ranking],
        -np.inf,
    )

    return ranking[rows, measured.argmax(axis=1)]


def score(
    picked: np.ndarray,
    target: np.ndarray,
    target_power: np.ndarray,
    shortlist_sizes: np.ndarray,
) -> dict[str, float]:
    """Exact and link-level metrics of the kept beams."""

    rows = np.arange(len(target))
    loss_db = 10.0 * np.log10(target_power.max(axis=1) / target_power[rows, picked])

    return {
        "exact_top1": float(np.mean(picked == target)),
        "macro_f1": float(f1_score(target, picked, average="macro", zero_division=0)),
        "macro_f1_support_ge5": float(f1_score(
            target, picked, average="macro", zero_division=0,
            labels=np.flatnonzero(np.bincount(target, minlength=BEAMS) >= 5),
        )),
        "within_noise_acc": float(np.mean(loss_db <= NOISE_TOLERANCE_DB)),
        "power_ratio": float(np.mean(target_power[rows, picked] / target_power.max(axis=1))),
        "mean_beams_measured": float(shortlist_sizes.mean()),
        "overhead_pct_of_full_sweep": float(100.0 * shortlist_sizes.mean() / BEAMS),
    }


def fixed_k_sizes(k: int, samples: int) -> np.ndarray:
    return np.full(samples, k)


def adaptive_sizes(probabilities_ranked: np.ndarray, tau: float) -> np.ndarray:
    """Smallest K whose cumulative ranked probability reaches tau (1..K_MAX)."""

    cumulative = np.cumsum(probabilities_ranked, axis=1)
    sizes = (cumulative < tau).sum(axis=1) + 1

    return np.clip(sizes, 1, K_MAX)


def sequence_bootstrap_ci(
    picked: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
) -> dict[str, list[float]]:
    """
    95% CIs for exact top-1 and macro-F1, resampling whole flight sequences.

    Samples inside a sequence are strongly correlated, so resampling
    samples would understate the uncertainty; the 35 sequences are the
    independent units.
    """

    rng = np.random.default_rng(seed)
    sequences = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in sequences}
    top1, macro = [], []

    for _ in range(BOOTSTRAP_RESAMPLES):
        index = np.concatenate([members[g] for g in rng.choice(sequences, len(sequences))])
        top1.append(np.mean(picked[index] == target[index]))
        macro.append(f1_score(target[index], picked[index], average="macro", zero_division=0))

    return {
        "exact_top1_ci95": [float(np.percentile(top1, 2.5)), float(np.percentile(top1, 97.5))],
        "macro_f1_ci95": [float(np.percentile(macro, 2.5)), float(np.percentile(macro, 97.5))],
    }


def nested_operating_point(
    settings: list,
    sizes_for: callable,
    ranking: np.ndarray,
    data: dict,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """
    For each fold, choose the cheapest setting meeting TARGET on both exact
    top-1 and macro-F1 over the other folds, then score the held-out fold.
    """

    picked_all = np.empty(len(data["target"]), dtype=int)
    sizes_all = np.empty(len(data["target"]), dtype=int)
    chosen = []

    for selection, held_out in folds:
        candidates = []
        for setting in settings:
            sizes = sizes_for(setting, selection)
            picked = measure_and_pick(ranking[selection], sizes, data["target_power"][selection])
            metrics = score(picked, data["target"][selection], data["target_power"][selection], sizes)
            if metrics["exact_top1"] >= TARGET and metrics["macro_f1"] >= TARGET:
                candidates.append((metrics["mean_beams_measured"], setting))

        setting = min(candidates)[1] if candidates else settings[-1]
        chosen.append(setting)

        sizes = sizes_for(setting, held_out)
        picked_all[held_out] = measure_and_pick(ranking[held_out], sizes, data["target_power"][held_out])
        sizes_all[held_out] = sizes

    return {
        "chosen_per_fold": chosen,
        **score(picked_all, data["target"], data["target_power"], sizes_all),
        **sequence_bootstrap_ci(picked_all, data["target"], data["groups"]),
    }


def main() -> None:
    """Score every shortlist method at every K and the nested operating points."""

    ceiling = noncausal_oracle_ceiling()
    print(f"Non-causal oracle ceiling (blind prediction): exact top-1 "
          f"{ceiling['exact_top1']:.3f}, macro-F1 {ceiling['macro_f1']:.3f}")

    curve_rows = []
    operating_points = []

    for horizon in HORIZONS:
        data = dict(np.load(RESULTS_DIR / f"beam_sequence_oof_h{horizon}.npz"))
        samples = len(data["target"])
        rankings = shortlist_rankings(data)

        # The same deterministic split the models were trained with; each
        # fold's selection set is the union of the other four test folds.
        test_folds = [test for _, test in GroupKFold(n_splits=FOLDS).split(
            data["target"], data["target"], data["groups"])]
        folds = [
            (np.concatenate([t for j, t in enumerate(test_folds) if j != i]), test_folds[i])
            for i in range(FOLDS)
        ]

        for method, ranking in rankings.items():
            for k in K_VALUES:
                sizes = fixed_k_sizes(k, samples)
                picked = measure_and_pick(ranking, sizes, data["target_power"])
                curve_rows.append({"horizon": horizon, "method": method, "shortlist": f"K={k}",
                                   **score(picked, data["target"], data["target_power"], sizes)})

            operating_points.append({
                "horizon": horizon, "method": method, "policy": "fixed K (nested CV)",
                **nested_operating_point(
                    list(K_VALUES), lambda k, index: fixed_k_sizes(k, len(index)),
                    ranking, data, folds,
                ),
            })

        ranked_probabilities = np.take_along_axis(
            softmax(data["gru_soft_target"]), rankings["gru_soft"], axis=1
        )
        for tau in TAU_VALUES:
            sizes = adaptive_sizes(ranked_probabilities, tau)
            picked = measure_and_pick(rankings["gru_soft"], sizes, data["target_power"])
            curve_rows.append({"horizon": horizon, "method": "gru_soft_adaptive", "shortlist": f"tau={tau}",
                               **score(picked, data["target"], data["target_power"], sizes)})

        operating_points.append({
            "horizon": horizon, "method": "gru_soft_adaptive", "policy": "adaptive tau (nested CV)",
            **nested_operating_point(
                list(TAU_VALUES), lambda tau, index: adaptive_sizes(ranked_probabilities[index], tau),
                rankings["gru_soft"], data, folds,
            ),
        })

    curves = pd.DataFrame(curve_rows)
    points = pd.DataFrame(operating_points)

    RESULTS_DIR.mkdir(exist_ok=True)
    curves.to_csv(RESULTS_DIR / "beam_training_overhead_curves.csv", index=False)
    (RESULTS_DIR / "beam_training_overhead.json").write_text(json.dumps({
        "target": TARGET,
        "noncausal_oracle_ceiling": ceiling,
        "operating_points": points.to_dict(orient="records"),
    }, indent=2, default=str))

    columns = ["exact_top1", "macro_f1", "macro_f1_support_ge5", "within_noise_acc", "power_ratio", "mean_beams_measured",
               "overhead_pct_of_full_sweep"]
    pd.set_option("display.width", 200)
    print("\nOperating points (nested CV):")
    print(points[["horizon", "method", "chosen_per_fold"] + columns].round(3).to_string(index=False))
    print("\n95% CIs (bootstrap over flight sequences):")
    for row in points.itertuples():
        print(f"  t+{row.horizon} {row.method:18s} top-1 {np.round(row.exact_top1_ci95, 3)}  "
              f"macro-F1 {np.round(row.macro_f1_ci95, 3)}")
    print("\nFixed-K curves (exact top-1 / macro-F1):")
    pivot = curves[curves["shortlist"].str.startswith("K=")].pivot_table(
        index=["horizon", "shortlist"], columns="method", values=["exact_top1", "macro_f1"]
    )
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
