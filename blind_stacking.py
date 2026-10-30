"""
Stacked blind beam predictor: several out-of-fold predictors, combined per fold.

Components, all out-of-fold over flights (GroupKFold over seq_index, the folds
of beam_sequence_model.py):

- the last measured sweep (persistence);
- a shifted template: the UAV's sub-beam peak position (parabolic fit around
  the argmax) is tracked over the recent sweeps, the last sweeps are aligned on
  it and averaged, and the average is shifted by the predicted peak motion;
  its three settings are chosen on the training folds;
- the GRU variants of beam_sequence_model.py (hard label, soft target,
  augmented ensemble) from each side-information run in results/<dir>/;
- extra trees and LightGBM on long-history features (8 peak positions and
  their steps, 4 sweeps, the lobe around the peak, side information), for the
  power-only and the GPS + UAV pixel inputs.

Each component's scores are z-scored per sample and summed with weights found
by coordinate ascent on the training folds (objective: within-noise accuracy
+ 0.5 x exact top-1), then applied unchanged to the held-out fold.

Two stacks are reported: power only (comparable with Table VII) and with all
sensing (GPS, camera UAV pixel). Metrics as in Table VII, plus the decisive /
near-tie split and per-flight tests against the GRU ensemble.

Needs results/{power_only,gps,uav_pixel,gps_uav_pixel}/beam_sequence_oof_h*.npz
(missing runs are skipped). Writes results/blind_stacking.{json,csv} and the stacked
scores to results/stacked_{power,all}/beam_sequence_oof_h*.npz for beam_training_overhead.py.
"""

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

from beam_prediction import (BEAMS, FOLDS, NOISE_TOLERANCE_DB, PROJECT_ROOT, SEED, link_metrics,
                             load_scenario23)
from beam_sequence_model import WINDOW
from blind_prediction_analysis import flight_bootstrap
from significance_tests import holm, paired_test

RESULTS = PROJECT_ROOT / "results"
RUNS = {"power only": ("power_only", "power"), "GPS": ("gps", "auto"),
        "UAV pixel": ("uav_pixel", "uav_pixel"), "GPS + UAV pixel": ("gps_uav_pixel", "gps+uav_pixel")}
GRU_KEYS = ("gru_hard_label", "gru_soft_target", "gru_soft_aug_ens")
TREE_RUNS = ("power only", "GPS + UAV pixel")   # trees are slow; fit them on the two extremes
SHIFT_GRID = [(fit, align, gain) for fit in (4, 8, 16) for align in (1, 2, 4) for gain in (0.25, 0.5, 0.75)]
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
SWEEPS = 3


def zscore(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / (x.std(axis=1, keepdims=True) + 1e-9)


def subbeam_peak(relative_db: np.ndarray) -> np.ndarray:
    """Peak position of every sweep with parabolic interpolation around the argmax."""

    i = relative_db.argmax(axis=1)
    rows = np.arange(len(i))
    inner = (i > 0) & (i < BEAMS - 1)
    a = relative_db[rows, np.clip(i - 1, 0, BEAMS - 1)]
    b = relative_db[rows, i]
    c = relative_db[rows, np.clip(i + 1, 0, BEAMS - 1)]
    curvature = a - 2 * b + c
    offset = np.where(inner & (curvature < 0), 0.5 * (a - c) / np.where(curvature < 0, curvature, -1), 0.0)
    return i + offset


def shift(row: np.ndarray, delta: float) -> np.ndarray:
    axis = np.arange(BEAMS, dtype=float)
    return np.interp(axis - delta, axis, row, left=row[0], right=row[-1])


def shifted_template(relative_db, peak, sequences, time_s, ends, horizon, fit, align, gain):
    """Causal template prediction of the sweep at t + horizon for every t in ends."""

    out = np.empty((len(ends), BEAMS))
    for n, t in enumerate(ends):
        start = t
        while start > 0 and sequences[start - 1] == sequences[t] and t - start < fit - 1:
            start -= 1
        index = np.arange(start, t + 1)
        velocity = np.polyfit(time_s[index] - time_s[t], peak[index], 1)[0] if len(index) >= 3 else 0.0
        first = max(start, t - align + 1)
        template = np.mean([shift(relative_db[k], peak[t] - peak[k]) for k in range(first, t + 1)], axis=0)
        out[n] = shift(template, gain * velocity * (time_s[t + horizon] - time_s[t]))
    return out


def tree_features(relative_db, peak, time_s, side, ends, horizon):
    past = [ends - k for k in range(WINDOW)]
    lobe = np.clip(np.rint(peak[ends]).astype(int)[:, None] + np.arange(-6, 7), 0, BEAMS - 1)
    columns = [relative_db[p] for p in past[:4]]
    columns += [np.stack([peak[p] for p in past], axis=1),
                np.stack([peak[past[k]] - peak[past[k + 1]] for k in range(WINDOW - 1)], axis=1),
                np.stack([time_s[past[k]] - time_s[past[k + 1]] for k in range(WINDOW - 1)], axis=1),
                (time_s[ends + horizon] - time_s[ends])[:, None],
                np.take_along_axis((relative_db[past[0]] + relative_db[past[1]]) / 2, lobe, axis=1)]
    if side is not None:
        columns += [side[p] for p in past[:SWEEPS]] + [side[past[0]] - side[past[SWEEPS - 1]]]
    return np.hstack(columns)


def tree_scores(features, labels, folds):
    """Out-of-fold log-probabilities of extra trees and LightGBM."""

    scores = {"extra trees": np.zeros((len(labels), BEAMS)), "LightGBM": np.zeros((len(labels), BEAMS))}
    for train, test in folds:
        trees = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features=0.3, n_jobs=-1,
                                     random_state=SEED)
        boost = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                                   n_jobs=-1, verbose=-1, random_state=SEED)
        for name, model in (("extra trees", trees), ("LightGBM", boost)):
            model.fit(features[train], labels[train])
            scores[name][np.ix_(test, model.classes_)] = model.predict_proba(features[test])
    return {name: np.log(s + 1e-4) for name, s in scores.items()}


def objective(scores, target, target_power) -> float:
    m = link_metrics(scores, target, target_power)
    return m["within_noise_acc"] + 0.5 * m["top1_acc"]


def fit_weights(components: dict, start: str, rows, target, target_power) -> dict:
    """Coordinate ascent over WEIGHTS on the given rows."""

    weights = {name: 0.0 for name in components}
    weights[start] = 1.0
    current = components[start][rows].copy()
    best = objective(current, target[rows], target_power[rows])
    for _ in range(3):
        changed = False
        for name in components:
            for value in WEIGHTS:
                trial = current + (value - weights[name]) * components[name][rows]
                score = objective(trial, target[rows], target_power[rows])
                if score > best + 1e-6:
                    best, current, weights[name], changed = score, trial, value, True
        if not changed:
            break
    return weights


def main() -> None:
    os.environ["SCENARIO23_FEATURES"] = "power"
    power, labels, sequences, time_s, _ = load_scenario23()
    relative_db = 10.0 * np.log10(power / power.max(axis=1, keepdims=True))
    peak = subbeam_peak(relative_db)
    sides = {}
    for name, (_, mode) in RUNS.items():
        if mode != "power":
            os.environ["SCENARIO23_FEATURES"] = mode
            sides[name] = load_scenario23()[4]
    rng = np.random.default_rng(SEED)

    report, rows, weights_log = {"noise_tolerance_db": NOISE_TOLERANCE_DB, "horizons": {}}, [], {}
    for horizon in (1, 3):
        base = dict(np.load(RESULTS / "power_only" / f"beam_sequence_oof_h{horizon}.npz"))
        target, target_power, groups = base["target"], base["target_power"], base["groups"]
        ends = np.array([t for t in range(WINDOW - 1, len(labels) - horizon)
                         if sequences[t - WINDOW + 1] == sequences[t + horizon]])
        assert (labels[ends + horizon] == target).all()
        folds = list(GroupKFold(n_splits=FOLDS).split(target, groups=groups))

        started = time.time()
        shifted = {setting: zscore(shifted_template(relative_db, peak, sequences, time_s, ends, horizon, *setting))
                   for setting in SHIFT_GRID}
        template = np.zeros_like(base["last_sweep_db"])
        for train, test in folds:
            setting = max(SHIFT_GRID, key=lambda s: objective(shifted[s][train], target[train], target_power[train]))
            template[test] = shifted[setting][test]
        print(f"t+{horizon}: shifted template {time.time() - started:.0f} s")

        components = {"power only": {"last sweep": zscore(base["last_sweep_db"]), "shifted template": template}}
        for run, (folder, _) in RUNS.items():
            path = RESULTS / folder / f"beam_sequence_oof_h{horizon}.npz"
            if not path.exists():
                print(f"  skip {run}: {path} missing")
                continue
            data = dict(np.load(path))
            assert (data["target"] == target).all()
            block = components.setdefault(run, {})
            block.update({f"{run}: {key}": zscore(data[key]) for key in GRU_KEYS})
            if run in TREE_RUNS:
                features = tree_features(relative_db, peak, time_s, sides.get(run), ends, horizon)
                started = time.time()
                block.update({f"{run}: {k}": zscore(v) for k, v in tree_scores(features, target, folds).items()})
                print(f"  {run}: trees {time.time() - started:.0f} s")

        stacks = {"Stacked, power only": ["power only"], "Stacked, all sensing": list(components)}
        candidates = {"Persistence": zscore(base["last_sweep_db"]),
                      "GRU ensemble (power only)": components["power only"]["power only: gru_soft_aug_ens"]}
        for stack, blocks in stacks.items():
            pool = {k: v for b in blocks for k, v in components[b].items()}
            scores = np.zeros_like(template)
            weights_log[f"{stack} t+{horizon}"] = []
            for train, test in folds:
                weights = fit_weights(pool, "power only: gru_soft_aug_ens", train, target, target_power)
                scores[test] = sum(w * pool[k][test] for k, w in weights.items() if w)
                weights_log[f"{stack} t+{horizon}"].append({k: w for k, w in weights.items() if w})
            candidates[stack] = scores
            # Saved with the power-only GRU scores so beam_training_overhead.py can shortlist from it.
            folder = RESULTS / ("stacked_power" if stack.endswith("power only") else "stacked_all")
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"beam_sequence_oof_h{horizon}.npz", **base, stacked=scores)
        for name, block in components.items():
            for key in block:
                if key.endswith(("LightGBM", "extra trees")) or key == "shifted template":
                    candidates[key] = block[key]

        top2 = np.sort(10 * np.log10(target_power), axis=1)[:, ::-1][:, :2]
        decisive = (top2[:, 0] - top2[:, 1]) > NOISE_TOLERANCE_DB
        reference_correct = candidates["GRU ensemble (power only)"].argmax(axis=1) == target
        flights = np.unique(groups)
        block = {}
        for name, scores in candidates.items():
            fold_rows = [link_metrics(scores[test], target[test], target_power[test]) for _, test in folds]
            metrics = {k: float(np.mean([r[k] for r in fold_rows])) for k in fold_rows[0]}
            correct = scores.argmax(axis=1) == target
            row = {"horizon": horizon, "model": name,
                   **{k: round(v, 4) for k, v in metrics.items()},
                   "exact_top1_ci95": flight_bootstrap(correct, groups, rng),
                   "exact_top1_decisive_pct": round(float(100 * correct[decisive].mean()), 1),
                   "exact_top1_near_tie_pct": round(float(100 * correct[~decisive].mean()), 1)}
            if name.startswith("Stacked"):
                per_flight = np.array([[correct[groups == f].mean(), reference_correct[groups == f].mean()]
                                       for f in flights]) * 100
                weights = np.array([(groups == f).sum() for f in flights], dtype=float)
                row["vs_gru_ensemble"] = paired_test(per_flight[:, 0], per_flight[:, 1], weights, rng)
            rows.append(row)
            block[name] = row
        report["horizons"][f"t+{horizon}"] = block

    stacked = [r for r in rows if "vs_gru_ensemble" in r]
    for row, p in zip(stacked, holm([r["vs_gru_ensemble"]["wilcoxon_p"] for r in stacked])):
        row["vs_gru_ensemble"]["holm_p"] = p
    report["weights_per_fold"] = weights_log
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "blind_stacking.json").write_text(json.dumps(report, indent=2, default=float))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]).to_csv(
        RESULTS / "blind_stacking.csv", index=False)
    for r in rows:
        test = r.get("vs_gru_ensemble")
        print(f"t+{r['horizon']} {r['model']:34s} noise {100 * r['within_noise_acc']:5.1f} "
              f"bal {100 * r['balanced_within_noise_acc']:5.1f} exact {100 * r['top1_acc']:5.1f} "
              f"{r['exact_top1_ci95']} F1 {r['macro_f1']:.3f} decisive {r['exact_top1_decisive_pct']} "
              f"near-tie {r['exact_top1_near_tie_pct']}"
              + (f"  vs GRU {test['difference']:+.2f} p {test['holm_p']:.2g} "
                 f"{test['wins']}/{test['ties']}/{test['losses']}" if test else ""))


if __name__ == "__main__":
    main()
