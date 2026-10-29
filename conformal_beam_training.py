"""Conformal beam shortlists with a coverage target on DeepSense 6G Scenario 23.

Reduced beam training measures a set of beams and keeps the strongest, so
it returns the exact optimal beam *if and only if* that beam is in the
measured set. Set coverage is therefore exact top-1 accuracy, and a
conformal set with coverage 1 - alpha targets exact top-1 >= 1 - alpha
directly, instead of searching for a K that happened to reach it.

Split conformal calibration (Vovk et al. 2005; Angelopoulos & Bates 2023),
grouped by flight: for each held-out fold, the threshold is computed on
the other four folds and applied unchanged to the held-out fold. Two
nonconformity scores:

- rank: the true beam's position in a ranking (GRU ensemble, or its
  tie-graph burst expansion). Gives a fixed set size per fold.
- aps: cumulative GRU probability up to and including the true beam
  (adaptive prediction sets, Romano et al. 2020). Gives a size that grows
  when the GRU is unsure.

Caveats, reported rather than hidden:
- Conformal coverage assumes exchangeable calibration and test samples.
  Samples inside a flight are correlated and flights differ, so the
  guarantee is approximate. The per-fold empirical coverage is reported.
- The out-of-fold GRU scores of the calibration folds come from models
  that saw the held-out fold in training (cross-conformal setting); the
  worst-case bound for such schemes is 1 - 2 alpha (Barber et al. 2021).
- The tie graph for each calibration fold is fitted without that fold and
  without the held-out fold; the held-out fold's graph uses all four
  calibration folds.

Latency: the whole set is measured in one burst (no report round trip),
so latency = set size x BEAM_MEASUREMENT_MS.

Run after beam_sequence_model.py:  python conformal_beam_training.py
Writes results/conformal_beam_training.json and .csv
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from beam_prediction import BEAMS, FOLDS
from beam_training_overhead import (
    BEAM_MEASUREMENT_MS,
    RESULTS_DIR,
    graph_expanded_ranking,
    measure_and_pick,
    score,
    sequence_bootstrap_ci,
    shortlist_rankings,
    softmax,
    tie_graph,
)

HORIZONS = (1, 3)
ALPHAS = (0.10, 0.05, 0.02, 0.01)
BURST_FIRST = (2, 4, 6)


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; inf if that exceeds n."""

    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def true_rank(ranking: np.ndarray, target: np.ndarray) -> np.ndarray:
    """1-based position of the target beam in each row's ranking."""

    return np.argmax(ranking == target[:, None], axis=1) + 1


def rank_set_sizes(threshold: float, samples: int) -> np.ndarray:
    return np.full(samples, int(min(threshold, BEAMS)))


def aps_scores(probabilities: np.ndarray, ranking: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cumulative probability of the ranked beams up to and including the target."""

    cumulative = np.cumsum(np.take_along_axis(probabilities, ranking, axis=1), axis=1)
    return cumulative[np.arange(len(target)), true_rank(ranking, target) - 1]


def aps_set_sizes(probabilities: np.ndarray, ranking: np.ndarray, threshold: float) -> np.ndarray:
    """Include rank k while the probability mass before it is below the threshold."""

    cumulative = np.cumsum(np.take_along_axis(probabilities, ranking, axis=1), axis=1)
    before = cumulative - np.take_along_axis(probabilities, ranking, axis=1)
    return np.clip((before < threshold).sum(axis=1), 1, BEAMS)


def evaluate(data: dict, test_folds: list[np.ndarray], alpha: float) -> list[dict]:
    """Conformal sets for each method at one alpha, scored on held-out folds."""

    target, power = data["target"], data["target_power"]
    gru_ranking = shortlist_rankings(data)["gru_aug_ens"]
    gru_probabilities = softmax(data["gru_soft_aug_ens"])
    rows = []

    methods = ["gru_aug_ens rank", "gru_aug_ens aps", "gru_aug_ens burst rank"]
    for method in methods:
        picked = np.empty(len(target), dtype=int)
        sizes = np.empty(len(target), dtype=int)
        per_fold, chosen = [], []

        for i, held_out in enumerate(test_folds):
            inner = [j for j in range(len(test_folds)) if j != i]
            calibration = np.concatenate([test_folds[j] for j in inner])

            if method == "gru_aug_ens rank":
                threshold = conformal_threshold(true_rank(gru_ranking[calibration], target[calibration]), alpha)
                ranking, size = gru_ranking[held_out], rank_set_sizes(threshold, len(held_out))
                chosen.append(int(min(threshold, BEAMS)))

            elif method == "gru_aug_ens aps":
                threshold = conformal_threshold(
                    aps_scores(gru_probabilities[calibration], gru_ranking[calibration], target[calibration]), alpha)
                ranking = gru_ranking[held_out]
                size = aps_set_sizes(gru_probabilities[held_out], ranking, threshold)
                chosen.append(round(threshold, 4))

            else:
                # Burst: choose how many GRU beams precede the tie partners
                # by the smallest calibrated set, all on calibration folds.
                best = None
                for first in BURST_FIRST:
                    calibration_ranks = []
                    for j in inner:
                        graph = tie_graph(power[np.concatenate([test_folds[k] for k in inner if k != j])])
                        expanded = graph_expanded_ranking(gru_ranking[test_folds[j]], graph, first)
                        calibration_ranks.append(true_rank(expanded, target[test_folds[j]]))
                    threshold = conformal_threshold(np.concatenate(calibration_ranks), alpha)
                    if best is None or threshold < best[0]:
                        best = (threshold, first)
                threshold, first = best
                ranking = graph_expanded_ranking(gru_ranking[held_out], tie_graph(power[calibration]), first)
                size = rank_set_sizes(threshold, len(held_out))
                chosen.append((first, int(min(threshold, BEAMS))))

            picked[held_out] = measure_and_pick(ranking, size, power[held_out])
            sizes[held_out] = size
            per_fold.append(float(np.mean(picked[held_out] == target[held_out])))

        rows.append({
            "method": method,
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "chosen_per_fold": chosen,
            "coverage_per_fold": [round(c, 4) for c in per_fold],
            "min_fold_coverage": min(per_fold),
            "set_size_p95": float(np.percentile(sizes, 95)),
            **score(picked, target, power, sizes, sizes * BEAM_MEASUREMENT_MS),
            **sequence_bootstrap_ci(picked, target, data["groups"]),
        })

    return rows


def main() -> None:
    """Calibrate and score conformal shortlists at every alpha and horizon."""

    all_rows = []
    for horizon in HORIZONS:
        data = dict(np.load(RESULTS_DIR / f"beam_sequence_oof_h{horizon}.npz"))
        test_folds = [test for _, test in GroupKFold(n_splits=FOLDS).split(
            data["target"], data["target"], data["groups"])]
        for alpha in ALPHAS:
            all_rows += [{"horizon": horizon, **row} for row in evaluate(data, test_folds, alpha)]

    table = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "conformal_beam_training.csv", index=False)
    (RESULTS_DIR / "conformal_beam_training.json").write_text(json.dumps({
        "beam_measurement_ms": BEAM_MEASUREMENT_MS,
        "rows": table.to_dict(orient="records"),
    }, indent=2, default=str))

    columns = ["horizon", "method", "target_coverage", "exact_top1", "min_fold_coverage", "macro_f1",
               "macro_f1_support_ge5", "mean_beams_measured", "set_size_p95", "latency_ms_mean"]
    pd.set_option("display.width", 220)
    print(table[columns].round(3).to_string(index=False))
    print("\nPer-fold coverage and chosen thresholds:")
    for row in table.itertuples():
        print(f"  t+{row.horizon} {row.method:24s} 1-a={row.target_coverage:.2f} "
              f"folds {row.coverage_per_fold} chosen {row.chosen_per_fold} "
              f"top-1 CI {np.round(row.exact_top1_ci95, 3)}")


if __name__ == "__main__":
    main()
