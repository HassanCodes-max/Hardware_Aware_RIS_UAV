"""
Where blind exact-label prediction fails, and a fused GRU + last-sweep predictor.

Blind exact top-1 on Scenario 23 is capped near 79% by measurement noise
(beam_training_overhead.noncausal_oracle_ceiling): in 75% of sweeps the best
and second-best beams are within the 0.25 dB noise floor, so which of them is
the "exact" label is decided by noise. This script

1. splits every held-out prediction of beam_sequence_model.py by how
   decisive the label is: the best beam of the target sweep leads the
   runner-up by more than NOISE_TOLERANCE_DB ("decisive"), or not ("near-tie"),
   and reports exact top-1 on each part with a 95% flight-bootstrap CI;
2. fuses the GRU ensemble's log-probabilities with the last measured sweep,
   score = z(log p) + w * z(last sweep dB), with w chosen by exact top-1 on
   the other four folds of flights (nested), and scores it like Table VII.

Reads results/<dir>/beam_sequence_oof_h{1,3}.npz (BEAM_RESULTS_DIR, default
results/), writes blind_prediction_analysis.json there.
"""

import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from beam_prediction import FOLDS, NOISE_TOLERANCE_DB, RESULTS_DIR, link_metrics

MODELS = {"persistence": "last_sweep_db", "GRU, hard label": "gru_hard_label",
          "GRU, soft target": "gru_soft_target", "GRU ensemble, augmented": "gru_soft_aug_ens"}
FUSION_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
RESAMPLES = 2000
SEED = 11


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


def flight_bootstrap(correct: np.ndarray, groups: np.ndarray, rng) -> list[float]:
    flights = np.unique(groups)
    hits = np.array([correct[groups == f].sum() for f in flights], dtype=float)
    counts = np.array([(groups == f).sum() for f in flights], dtype=float)
    draws = rng.integers(0, len(flights), size=(RESAMPLES, len(flights)))
    rates = hits[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1)
    return [round(float(100 * np.percentile(rates, q)), 1) for q in (2.5, 97.5)]


def fold_mean_metrics(scores: np.ndarray, data: dict, folds) -> dict:
    rows = [link_metrics(scores[test], data["target"][test], data["target_power"][test]) for _, test in folds]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main() -> None:
    rng = np.random.default_rng(SEED)
    report = {"noise_tolerance_db": NOISE_TOLERANCE_DB, "horizons": {}}
    for horizon in (1, 3):
        data = dict(np.load(RESULTS_DIR / f"beam_sequence_oof_h{horizon}.npz"))
        target, groups = data["target"], data["groups"]
        folds = list(GroupKFold(n_splits=FOLDS).split(target, groups=groups))
        top2 = np.sort(10 * np.log10(data["target_power"]), axis=1)[:, ::-1][:, :2]
        decisive = (top2[:, 0] - top2[:, 1]) > NOISE_TOLERANCE_DB

        # Fused predictor, weight chosen on the other folds (exact top-1).
        log_p = zscore(data["gru_soft_aug_ens"])   # the ensemble stores log-probabilities
        last = zscore(data["last_sweep_db"])
        fused = np.zeros_like(log_p)
        weights = []
        for train, test in folds:
            w = max(FUSION_WEIGHTS, key=lambda v: np.mean(
                (log_p[train] + v * last[train]).argmax(axis=1) == target[train]))
            weights.append(w)
            fused[test] = log_p[test] + w * last[test]

        rows = {}
        for name, scores in list({n: data[k] for n, k in MODELS.items()}.items()) + [
                ("GRU ensemble + last sweep, fused", fused)]:
            picked = scores.argmax(axis=1)
            correct = picked == target
            rows[name] = {
                **{k: round(v, 4) for k, v in fold_mean_metrics(scores, data, folds).items()},
                "exact_top1_decisive_pct": round(float(100 * correct[decisive].mean()), 1),
                "exact_top1_decisive_ci95": flight_bootstrap(correct[decisive], groups[decisive], rng),
                "exact_top1_near_tie_pct": round(float(100 * correct[~decisive].mean()), 1),
                "exact_top1_near_tie_ci95": flight_bootstrap(correct[~decisive], groups[~decisive], rng),
                "macro_f1_decisive": round(float(f1_score(target[decisive], picked[decisive],
                                                          average="macro", zero_division=0)), 3),
            }
        report["horizons"][f"t+{horizon}"] = {
            "decisive_share_pct": round(float(100 * decisive.mean()), 1),
            "fusion_weight_per_fold": weights,
            "models": rows,
        }

    (RESULTS_DIR / "blind_prediction_analysis.json").write_text(json.dumps(report, indent=2))
    for h, block in report["horizons"].items():
        print(f"\n{h}: decisive share {block['decisive_share_pct']}%, fusion weights {block['fusion_weight_per_fold']}")
        for name, r in block["models"].items():
            print(f"  {name:34s} exact {100 * r['top1_acc']:5.1f}  macro-F1 {r['macro_f1']:.3f}  "
                  f"decisive {r['exact_top1_decisive_pct']:5.1f} {r['exact_top1_decisive_ci95']}  "
                  f"near-tie {r['exact_top1_near_tie_pct']:5.1f}")


if __name__ == "__main__":
    main()
