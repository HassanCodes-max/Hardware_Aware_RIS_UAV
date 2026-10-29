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
- gru_aug_ens / gru_aug_ens+sweep: the same for the augmented 3-seed
  ensemble (its scores are already averaged log-probabilities);
- gru_soft+sweep: z-scored GRU log-probabilities plus the z-scored last
  sweep, equally weighted. The GRU carries motion trends, the last sweep
  carries the current fine structure, and each corrects the other.

Tried and rejected: logit adjustment for the class imbalance (Menon et
al., 2021). The beam distribution differs too much between flights for a
training-fold prior to transfer; it lowered every metric.

Adaptive shortlist: the smallest set of GRU beams whose softmax
probability reaches tau (at most K_MAX), so confident cases measure fewer
beams.

Two-round search with a learned tie graph (the 3GPP P1/P2 split: a
coarse set, then a refinement set). Round 1 measures the first K1 beams
of a ranking. Round 2 measures the M beams that, in the training flights,
were most often within TIE_DB of the best when round 1's winner was the
best. The strong beams of this codebook are not contiguous (beams 27, 28,
32, 33 and 38 are often within 0.1 dB of each other), so a window around
the last beam misses them and the tie graph does not. If the best
measured power is still more than DROP_DB below the last sweep's peak,
the beam is treated as lost and the remaining beams are swept (fallback).
The graph is always fitted on folds other than the ones it is scored on,
including inside the nested selection.

Single burst (graph-expanded): the first K ranked beams plus the
tie-graph partners of the two top-ranked beams, all measured in one
burst. It brings the tie graph's coverage without the report round trip
of a second round.

Latency: each method's beam-training time on the critical path, under
5G NR FR2 timing (latency_ms). Inference is not on the critical path:
the shortlist for t + h is computed from sweep t, during the ~140 ms
before t + h.

Operating points are chosen with nested cross-validation: for each
held-out fold, the setting is the lowest-latency one that meets the
target on the other four folds, then applied to the held-out fold.

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
TARGET = 0.95
BOOTSTRAP_RESAMPLES = 500

# Two-round search grid: (round-1 beams, round-2 beams, fallback drop in dB).
TIE_DB = 1.0
TWO_ROUND_GRID = [
    (k1, m, drop)
    for k1 in (2, 3, 4, 6, 8, 10, 12)
    for m in (0, 2, 4, 6, 8, 10)
    for drop in (np.inf, 3.0)
]
TWO_ROUND_METHODS = ("last_sweep", "gru_aug_ens", "gru_aug_ens+sweep")

# Single-burst graph-expanded shortlist: (ranked beams, tie partners added).
BURST_GRID = [(first, extra) for first in (2, 3, 4, 6, 8, 10) for extra in (0, 2, 4, 6, 8, 10)]

# 5G NR FR2, 120 kHz SCS (SSB case D): up to 64 SSB beams in one 5 ms
# half-frame (TS 38.213 Sec. 4.1), so ~78 us per beam measurement.
# A beam report and the next measurement grant are assumed to take 8
# slots (1 ms). RIS switching is the same for every method and excluded.
BEAM_MEASUREMENT_MS = 5.0 / 64
REPORT_ROUND_TRIP_MS = 1.0


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
        "gru_aug_ens": np.argsort(-data["gru_soft_aug_ens"], axis=1),
        "gru_aug_ens+sweep": np.argsort(-(zscore(data["gru_soft_aug_ens"]) + zscore(last)), axis=1),
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


def tie_graph(target_power: np.ndarray) -> np.ndarray:
    """
    Rank, for each beam b, the other beams by how often they were within
    TIE_DB of the best when b was the best (fitted on training samples).

    Returns (64, 64): row b lists candidate beams, most frequent tie first;
    never-seen pairs are ordered by beam-index distance.
    """

    power_db = 10.0 * np.log10(target_power)
    close = power_db >= power_db.max(axis=1, keepdims=True) - TIE_DB
    counts = np.zeros((BEAMS, BEAMS))
    np.add.at(counts, power_db.argmax(axis=1), close.astype(float))

    beams = np.arange(BEAMS)
    distance = np.abs(beams[None, :] - beams[:, None])
    counts[beams, beams] = -np.inf

    return np.lexsort((distance, -counts), axis=1)


def two_round_pick(
    ranking: np.ndarray,
    first_round: int,
    second_round: int,
    drop_db: float,
    target_power: np.ndarray,
    graph: np.ndarray,
    last_peak_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Round 1: the first `first_round` ranked beams. Round 2: up to
    `second_round` unmeasured tie-graph candidates of round 1's winner.
    Fallback: if the best measured beam is more than `drop_db` below the
    last sweep's peak, measure every remaining beam.

    Returns the kept beam, beams measured and latency (ms) per sample.
    """

    samples = len(ranking)
    rows = np.arange(samples)
    measured = np.zeros((samples, BEAMS), dtype=bool)
    measured[rows[:, None], ranking[:, :first_round]] = True
    latency = np.full(samples, first_round * BEAM_MEASUREMENT_MS)

    if second_round > 0:
        winner = np.where(measured, target_power, -np.inf).argmax(axis=1)
        added = np.zeros(samples, dtype=int)
        for column in range(BEAMS - 1):
            if (added >= second_round).all():
                break
            candidate = graph[winner, column]
            new = ~measured[rows, candidate] & (added < second_round)
            measured[rows[new], candidate[new]] = True
            added += new
        latency += REPORT_ROUND_TRIP_MS + added * BEAM_MEASUREMENT_MS

    best = np.where(measured, target_power, -np.inf).max(axis=1)
    lost = 10.0 * np.log10(best / last_peak_power) < -drop_db
    latency[lost] += REPORT_ROUND_TRIP_MS + (BEAMS - measured[lost].sum(axis=1)) * BEAM_MEASUREMENT_MS
    measured[lost] = True

    return np.where(measured, target_power, -np.inf).argmax(axis=1), measured.sum(axis=1), latency


def graph_expanded_ranking(
    ranking: np.ndarray,
    graph: np.ndarray,
    first: int,
) -> np.ndarray:
    """
    Single-burst shortlist with no feedback round trip: the first `first`
    ranked beams, then the tie-graph partners of the two top-ranked beams
    (interleaved), then the rest of the ranking. Duplicates keep their
    first position.
    """

    rows = np.arange(len(ranking))
    partners = np.stack([graph[ranking[:, 0]], graph[ranking[:, 1]]], axis=2).reshape(len(ranking), -1)
    candidates = np.concatenate([ranking[:, :first], partners, ranking], axis=1)

    position = np.full((len(ranking), BEAMS), candidates.shape[1])
    for column in range(candidates.shape[1] - 1, -1, -1):
        position[rows, candidates[:, column]] = column

    return np.argsort(position, axis=1, kind="stable")


def last_peak_power(horizon: int, target: np.ndarray) -> np.ndarray:
    """Peak power of the last measured sweep for each window (beam_sequence_model's indexing)."""

    from beam_sequence_model import WINDOW

    power, labels, sequences, _, _ = load_scenario23()
    ends = np.array([
        t for t in range(WINDOW - 1, len(labels) - horizon)
        if sequences[t - WINDOW + 1] == sequences[t + horizon]
    ])
    assert np.array_equal(labels[ends + horizon], target), "window indexing changed"

    return power[ends].max(axis=1)


def score(
    picked: np.ndarray,
    target: np.ndarray,
    target_power: np.ndarray,
    shortlist_sizes: np.ndarray,
    latency_ms: np.ndarray | None = None,
) -> dict[str, float]:
    """Exact and link-level metrics of the kept beams."""

    rows = np.arange(len(target))
    loss_db = 10.0 * np.log10(target_power.max(axis=1) / target_power[rows, picked])
    if latency_ms is None:
        latency_ms = shortlist_sizes * BEAM_MEASUREMENT_MS

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
        "latency_ms_mean": float(latency_ms.mean()),
        "latency_ms_p95": float(np.percentile(latency_ms, 95)),
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
    supported = np.flatnonzero(np.bincount(target, minlength=BEAMS) >= 5)
    top1, macro, macro_supported = [], [], []

    for _ in range(BOOTSTRAP_RESAMPLES):
        index = np.concatenate([members[g] for g in rng.choice(sequences, len(sequences))])
        top1.append(np.mean(picked[index] == target[index]))
        macro.append(f1_score(target[index], picked[index], average="macro", zero_division=0))
        macro_supported.append(f1_score(target[index], picked[index], average="macro",
                                        zero_division=0, labels=supported))

    def interval(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {
        "exact_top1_ci95": interval(top1),
        "macro_f1_ci95": interval(macro),
        "macro_f1_support_ge5_ci95": interval(macro_supported),
    }


def nested_operating_point(
    settings: list,
    policy: callable,
    data: dict,
    test_folds: list[np.ndarray],
) -> dict:
    """
    For each fold, choose the lowest-latency setting meeting TARGET on both
    exact top-1 and all-class macro-F1 over the other folds (or, if none
    does, the one with the highest macro-F1), then score the held-out fold.

    `policy(setting, index, graph)` returns (kept beam, beams measured,
    latency ms) for the samples in `index`. The tie graph is always fitted
    on folds other than the ones scored: during selection, each selection
    fold uses a graph from the remaining three; the held-out fold uses a
    graph from all four selection folds.
    """

    picked_all = np.empty(len(data["target"]), dtype=int)
    sizes_all = np.empty(len(data["target"]), dtype=int)
    latency_all = np.empty(len(data["target"]))
    chosen = []

    for i, held_out in enumerate(test_folds):
        inner = [j for j in range(len(test_folds)) if j != i]
        selection = np.concatenate([test_folds[j] for j in inner])
        inner_graphs = {
            j: tie_graph(data["target_power"][np.concatenate([test_folds[k] for k in inner if k != j])])
            for j in inner
        }
        graph = tie_graph(data["target_power"][selection])
        candidates, fallback = [], []
        for setting in settings:
            parts = [policy(setting, test_folds[j], inner_graphs[j]) for j in inner]
            picked, sizes, latency = (np.concatenate(part) for part in zip(*parts))
            metrics = score(picked, data["target"][selection], data["target_power"][selection], sizes, latency)
            fallback.append((-metrics["macro_f1"], metrics["latency_ms_mean"], setting))
            if metrics["exact_top1"] >= TARGET and metrics["macro_f1"] >= TARGET:
                candidates.append((metrics["latency_ms_mean"], metrics["mean_beams_measured"], setting))

        setting = min(candidates)[-1] if candidates else min(fallback)[-1]
        chosen.append(setting)
        picked_all[held_out], sizes_all[held_out], latency_all[held_out] = policy(setting, held_out, graph)

    return {
        "chosen_per_fold": chosen,
        **score(picked_all, data["target"], data["target_power"], sizes_all, latency_all),
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
        data["last_peak_power"] = last_peak_power(horizon, data["target"])
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

        def single_round(ranking: np.ndarray, sizes_for: callable) -> callable:
            def policy(setting, index, graph):
                sizes = sizes_for(setting, index)
                picked = measure_and_pick(ranking[index], sizes, data["target_power"][index])
                return picked, sizes, sizes * BEAM_MEASUREMENT_MS
            return policy

        def two_round(ranking: np.ndarray) -> callable:
            def policy(setting, index, graph):
                return two_round_pick(ranking[index], *setting, data["target_power"][index],
                                      graph, data["last_peak_power"][index])
            return policy

        for method, ranking in rankings.items():
            for k in K_VALUES:
                sizes = fixed_k_sizes(k, samples)
                picked = measure_and_pick(ranking, sizes, data["target_power"])
                curve_rows.append({"horizon": horizon, "method": method, "shortlist": f"K={k}",
                                   **score(picked, data["target"], data["target_power"], sizes)})

            operating_points.append({
                "horizon": horizon, "method": method, "policy": "fixed K (nested CV)",
                **nested_operating_point(
                    list(K_VALUES),
                    single_round(ranking, lambda k, index: fixed_k_sizes(k, len(index))),
                    data, test_folds,
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
                list(TAU_VALUES),
                single_round(rankings["gru_soft"],
                             lambda tau, index: adaptive_sizes(ranked_probabilities[index], tau)),
                data, test_folds,
            ),
        })

        # Two-round search. Curves use, for each fold, a tie graph fitted on
        # the other four folds, so every curve point is out-of-fold too.
        fold_graphs = [tie_graph(data["target_power"][selection]) for selection, _ in folds]
        for method in TWO_ROUND_METHODS:
            policy = two_round(rankings[method])
            for setting in TWO_ROUND_GRID:
                picked = np.empty(samples, dtype=int)
                sizes = np.empty(samples, dtype=int)
                latency = np.empty(samples)
                for (_, held_out), graph in zip(folds, fold_graphs):
                    picked[held_out], sizes[held_out], latency[held_out] = policy(setting, held_out, graph)
                k1, m, drop = setting
                curve_rows.append({"horizon": horizon, "method": f"{method} two-round",
                                   "shortlist": f"K1={k1},M={m},drop={drop}",
                                   **score(picked, data["target"], data["target_power"], sizes, latency)})

            operating_points.append({
                "horizon": horizon, "method": f"{method} two-round", "policy": "K1, M, drop (nested CV)",
                **nested_operating_point(TWO_ROUND_GRID, policy, data, test_folds),
            })

        def burst(ranking: np.ndarray) -> callable:
            def policy(setting, index, graph):
                first, extra = setting
                sizes = fixed_k_sizes(first + extra, len(index))
                expanded = graph_expanded_ranking(ranking[index], graph, first)
                return (measure_and_pick(expanded, sizes, data["target_power"][index]),
                        sizes, sizes * BEAM_MEASUREMENT_MS)
            return policy

        for method in TWO_ROUND_METHODS:
            policy = burst(rankings[method])
            for setting in BURST_GRID:
                picked = np.empty(samples, dtype=int)
                sizes = np.empty(samples, dtype=int)
                latency = np.empty(samples)
                for (_, held_out), graph in zip(folds, fold_graphs):
                    picked[held_out], sizes[held_out], latency[held_out] = policy(setting, held_out, graph)
                curve_rows.append({"horizon": horizon, "method": f"{method} burst",
                                   "shortlist": f"K={setting[0]}+{setting[1]}",
                                   **score(picked, data["target"], data["target_power"], sizes, latency)})

            operating_points.append({
                "horizon": horizon, "method": f"{method} burst", "policy": "K, partners (nested CV)",
                **nested_operating_point(BURST_GRID, policy, data, test_folds),
            })

    curves = pd.DataFrame(curve_rows)
    points = pd.DataFrame(operating_points)

    RESULTS_DIR.mkdir(exist_ok=True)
    curves.to_csv(RESULTS_DIR / "beam_training_overhead_curves.csv", index=False)
    (RESULTS_DIR / "beam_training_overhead.json").write_text(json.dumps({
        "target": TARGET,
        "noncausal_oracle_ceiling": ceiling,
        "latency_model": {"beam_measurement_ms": BEAM_MEASUREMENT_MS,
                          "report_round_trip_ms": REPORT_ROUND_TRIP_MS,
                          "full_sweep_ms": BEAMS * BEAM_MEASUREMENT_MS},
        "operating_points": points.to_dict(orient="records"),
    }, indent=2, default=str))

    columns = ["exact_top1", "macro_f1", "macro_f1_support_ge5", "within_noise_acc", "power_ratio", "mean_beams_measured",
               "latency_ms_mean", "latency_ms_p95"]
    pd.set_option("display.width", 200)
    print("\nOperating points (nested CV):")
    print(points[["horizon", "method", "chosen_per_fold"] + columns].round(3).to_string(index=False))
    print("\n95% CIs (bootstrap over flight sequences):")
    for row in points.itertuples():
        print(f"  t+{row.horizon} {row.method:28s} top-1 {np.round(row.exact_top1_ci95, 3)}  "
              f"macro-F1 {np.round(row.macro_f1_ci95, 3)}  macro-F1 (n>=5) {np.round(row.macro_f1_support_ge5_ci95, 3)}")
    print("\nFixed-K curves (exact top-1 / macro-F1):")
    pivot = curves[curves["shortlist"].str.startswith("K=")].pivot_table(
        index=["horizon", "shortlist"], columns="method", values=["exact_top1", "macro_f1"]
    )
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
