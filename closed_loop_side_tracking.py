"""
Closed-loop beam tracking aided by the camera or GPS (DeepSense 6G Scenario 23).

closed_loop_beam_tracking.py spends most of its beam-training time on
periodic full sweeps that catch the beam when it moves away from the
local window. Here a side-information predictor proposes extra
candidates at every step instead: a random forest that maps the sensing
of the previous step (t - 1) to the best beam at t, from

- camera: the UAV's pixel position and contrast in the BS camera frame
  (camera_beam_prediction.py --locate; no labels involved in locating);
- GPS: the UAV kinematics (east/north from the BS, bearing, speed, ...);
- camera + GPS: both.

Using the previous step's frame keeps the predictor causal and leaves one
sweep interval (~142 ms) for the 54 ms of image processing. The
predictor and the tie graph are fitted on the other folds' flights only
(GroupKFold(5) over seq_index). Policies, metrics and the latency model
are those of closed_loop_beam_tracking.py; each side-aided policy is
compared per flight with the tie-graph tracker (paired bootstrap CI,
Wilcoxon, Holm).

Run:  python closed_loop_side_tracking.py   (needs the full Scenario 23 download)
Writes results/closed_loop_side_tracking{.json,.csv,_flights.csv}
"""

import json

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from beam_prediction import BEAMS, DATA_DIR, FOLDS, RESULTS_DIR, SEED, load_scenario23
from beam_training_overhead import tie_graph
from camera_beam_prediction import uav_pixel_features
from beam_training_overhead import BEAM_MEASUREMENT_MS, REPORT_ROUND_TRIP_MS
from closed_loop_beam_tracking import (GENIE, NOMINAL_SNR_DB, Policy, effective_se, flight_record,
                                       run_flight, summarise_flights)
from public_data_adapter import load_scenario23_kinematics
from significance_tests import holm, paired_test

TRACKER = Policy("tracker: window +/-1 + 4 tie partners, full every 8", 8, window=1, partners=4)
SIDE_POLICIES = {
    "camera": [
        Policy("camera top-2 + tracker, full every 8", 8, window=1, partners=4, side=2),
        Policy("camera top-2 + tracker, loss-triggered only", None, window=1, partners=4, side=2),
        Policy("camera top-4 + tracker, loss-triggered only", None, window=1, partners=4, side=4),
        Policy("camera top-4 + window +/-1, loss-triggered only", None, window=1, side=4),
    ],
    "GPS": [
        Policy("GPS top-2 + tracker, loss-triggered only", None, window=1, partners=4, side=2),
        Policy("GPS top-4 + tracker, loss-triggered only", None, window=1, partners=4, side=4),
    ],
    "camera + GPS": [
        Policy("camera + GPS top-2 + tracker, loss-triggered only", None, window=1, partners=4, side=2),
        Policy("camera + GPS top-4 + tracker, loss-triggered only", None, window=1, partners=4, side=4),
    ],
}
# Stale GPS: the UAV's report arrives d steps late (~142 ms per step); the camera,
# local to the BS, stays one frame old.
GPS_LAGS = (2, 4, 8)
for lag in GPS_LAGS:
    SIDE_POLICIES[f"GPS lag {lag}"] = [Policy(f"GPS top-4 + tracker, loss-triggered only, GPS {lag} steps old",
                                              None, window=1, partners=4, side=4)]
    SIDE_POLICIES[f"camera + GPS lag {lag}"] = [Policy(
        f"camera + GPS top-4 + tracker, loss-triggered only, GPS {lag} steps old",
        None, window=1, partners=4, side=4)]
LATENCY_POLICIES = ("tracker: window +/-1 + 4 tie partners, full every 8",
                    "camera + GPS top-4 + tracker, loss-triggered only", "full sweep every step")
BEAM_TIMES_MS = (BEAM_MEASUREMENT_MS / 2, BEAM_MEASUREMENT_MS, 2 * BEAM_MEASUREMENT_MS)
ROUND_TRIPS_MS = (0.5, REPORT_ROUND_TRIP_MS, 2.0)

REFERENCE_POLICIES = [
    TRACKER,
    Policy("tracker, loss-triggered only", None, window=1, partners=4),
    Policy("full sweep every step", 1),
]


def lagged(previous: np.ndarray, steps: int) -> np.ndarray:
    """Index of the sample `steps` earlier in the same flight, -1 if none."""

    index = np.arange(len(previous))
    for _ in range(steps):
        index = np.where(index >= 0, previous[np.maximum(index, 0)], -1)
    return index


def side_rankings(features: np.ndarray, usable: np.ndarray, labels: np.ndarray,
                  train: np.ndarray) -> np.ndarray:
    """(N, 64) beam ranking per decision; features[i] is what is known when deciding sample i."""

    fit = train[usable[train]]
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    model.fit(features[fit], labels[fit])
    scores = np.zeros((len(labels), BEAMS))
    rows = np.flatnonzero(usable)
    scores[np.ix_(rows, model.classes_)] = model.predict_proba(features[rows])
    return np.argsort(-scores, axis=1)


def main() -> None:
    power, labels, sequences, time_s, _ = load_scenario23()
    manifest = pd.read_csv(DATA_DIR / "scenario23.csv")
    gps = load_scenario23_kinematics(DATA_DIR, manifest).to_numpy(float)
    camera = uav_pixel_features()

    # Index of the previous sample of the same flight (time order), -1 for a flight's first sample.
    previous = np.full(len(labels), -1)
    for sequence in np.unique(sequences):
        index = np.flatnonzero(sequences == sequence)
        index = index[np.argsort(time_s[index], kind="stable")]
        previous[index[1:]] = index[:-1]

    # What each decision knows: the camera frame one step old, the GPS `lag` steps old.
    one = lagged(previous, 1)
    side_features = {"camera": (camera[one], one >= 0), "GPS": (gps[one], one >= 0),
                     "camera + GPS": (np.hstack([camera[one], gps[one]]), one >= 0)}
    for lag in GPS_LAGS:
        old = lagged(previous, lag)
        side_features[f"GPS lag {lag}"] = (gps[old], old >= 0)
        side_features[f"camera + GPS lag {lag}"] = (np.hstack([camera[one], gps[old]]), old >= 0)
    kept_traces = defaultdict(list)   # policy -> [(power, time, trace)] for the latency sensitivity

    flight_rows = []
    decisions = defaultdict(lambda: ([], []))   # policy -> (best beams, chosen beams), all steps
    for train, test in GroupKFold(n_splits=FOLDS).split(power, groups=sequences):
        graph = tie_graph(power[train])
        rankings = {name: side_rankings(f, usable, labels, train)
                    for name, (f, usable) in side_features.items()}
        for sequence in np.unique(sequences[test]):
            index = np.flatnonzero(sequences == sequence)
            index = index[np.argsort(time_s[index], kind="stable")]
            flight_power, flight_time = power[index], time_s[index]
            if len(index) < 2 or np.any(np.diff(flight_time) <= 0):
                continue
            for policy in REFERENCE_POLICIES + [GENIE]:
                trace = run_flight(flight_power, policy, graph)
                flight_rows.append(flight_record(policy, int(sequence), flight_power, flight_time, trace))
                decisions[policy.name][0].append(flight_power.argmax(axis=1))
                decisions[policy.name][1].append(trace["chosen"])
                if policy.name in LATENCY_POLICIES:
                    kept_traces[policy.name].append((flight_power, flight_time, trace))
            for name, policies in SIDE_POLICIES.items():
                for policy in policies:
                    trace = run_flight(flight_power, policy, graph, rankings[name][index])
                    if policy.name in LATENCY_POLICIES:
                        kept_traces[policy.name].append((flight_power, flight_time, trace))
                    decisions[policy.name][0].append(flight_power.argmax(axis=1))
                    decisions[policy.name][1].append(trace["chosen"])
                    flight_rows.append(flight_record(policy, int(sequence), flight_power, flight_time,
                                                     trace))
            print(f"flight {sequence} done")

    flights = pd.DataFrame(flight_rows)
    flights["se_10dB"] = flights["se_10dB_sum"] / flights["intervals"]
    summary = summarise_flights(flights.drop(columns="se_10dB"))

    # Exact-label macro-F1 over all decisions of each policy: all classes, and classes
    # (best beams) with at least five steps, as for the open-loop study.
    for policy, (best, chosen) in decisions.items():
        best, chosen = np.concatenate(best), np.concatenate(chosen)
        counts = np.bincount(best, minlength=BEAMS)
        summary.loc[policy, "macro_f1"] = f1_score(best, chosen, average="macro", zero_division=0)
        summary.loc[policy, "macro_f1_support_ge5"] = f1_score(
            best, chosen, labels=np.flatnonzero(counts >= 5), average="macro", zero_division=0)
        summary.loc[policy, "classes"] = int((counts > 0).sum())

    # Latency-model sensitivity: per-beam time and report round trip, recomputed from each
    # step's beam count and its loss-triggered fallbacks (one round trip each).
    genie_rows = flights[flights["policy"] == GENIE.name]
    genie_se = genie_rows["se_10dB_sum"].sum() / genie_rows["intervals"].sum()
    latency_sensitivity = []
    for beam_ms in BEAM_TIMES_MS:
        for round_trip_ms in ROUND_TRIPS_MS:
            for policy in LATENCY_POLICIES:
                latencies, se_sum, intervals = [], 0.0, 0
                for flight_power, flight_time, trace in kept_traces[policy]:
                    trips = np.rint((trace["latency"] - trace["beams"] * BEAM_MEASUREMENT_MS)
                                    / REPORT_ROUND_TRIP_MS)
                    latency = trace["beams"] * beam_ms + trips * round_trip_ms
                    latencies.append(latency)
                    se = effective_se(flight_power, trace["chosen"], latency, np.diff(flight_time), 10.0)
                    se_sum, intervals = se_sum + se.sum(), intervals + len(se)
                latency_sensitivity.append({
                    "beam_ms": round(beam_ms, 4), "round_trip_ms": round_trip_ms, "policy": policy,
                    "latency_ms_mean": round(float(np.mean(np.concatenate(latencies))), 4),
                    "se_10dB_pct_of_genie": round(float(100 * se_sum / intervals / genie_se), 3)})

    # Per-flight tests against the tie-graph tracker.
    rng = np.random.default_rng(7)
    genie = flights[flights["policy"] == GENIE.name].set_index("sequence")["se_10dB"]
    tests = []
    reference = flights[flights["policy"] == TRACKER.name].set_index("sequence")
    for policy in [p for ps in SIDE_POLICIES.values() for p in ps] + REFERENCE_POLICIES[1:]:
        rows = flights[flights["policy"] == policy.name].set_index("sequence").loc[reference.index]
        for metric, scale in (("exact_top1", 100.0), ("latency_ms_mean", 1.0), ("se_10dB", None)):
            if scale is None:
                a = 100.0 * rows[metric] / genie.loc[rows.index]
                b = 100.0 * reference[metric] / genie.loc[reference.index]
            else:
                a, b = scale * rows[metric], scale * reference[metric]
            tests.append({"policy": policy.name, "metric": metric,
                          **paired_test(a.to_numpy(), b.to_numpy(), reference["steps"].to_numpy(float),
                                        rng)})
    tests = pd.DataFrame(tests)
    for metric, group in tests.groupby("metric").groups.items():
        tests.loc[group, "holm_p"] = holm(tests.loc[group, "wilcoxon_p"].tolist())

    RESULTS_DIR.mkdir(exist_ok=True)
    flights.to_csv(RESULTS_DIR / "closed_loop_side_tracking_flights.csv", index=False)
    summary.round(5).to_csv(RESULTS_DIR / "closed_loop_side_tracking.csv")
    (RESULTS_DIR / "closed_loop_side_tracking.json").write_text(json.dumps({
        "split": f"GroupKFold({FOLDS}) over seq_index; tie graph and side predictors fitted on other folds",
        "side_predictor": "RandomForest(300, min_samples_leaf=2): side features at t-1 -> best beam at t",
        "nominal_snr_db": list(NOMINAL_SNR_DB),
        "summary": summary.round(5).reset_index().to_dict(orient="records"),
        "tests_vs_tracker": tests.round(6).to_dict(orient="records"),
        "latency_sensitivity": latency_sensitivity,
    }, indent=2))

    pd.set_option("display.width", 250)
    columns = ["exact_top1", "macro_f1", "macro_f1_support_ge5", "within_noise", "beams_mean", "latency_ms_mean", "latency_ms_p95_worst_flight",
               "full_sweep_fraction", "se_10dB_pct_of_genie"]
    print(summary[columns].round(4).to_string())
    print(tests.round(4).to_string())


if __name__ == "__main__":
    main()
