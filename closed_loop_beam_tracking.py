"""Closed-loop beam tracking and effective throughput on DeepSense 6G Scenario 23.

beam_training_overhead.py scores reduced beam training open-loop: the
shortlist for t + h is built from the *full* 64-beam sweeps up to t. A
system that measures only K beams per step never has those full sweeps,
so those numbers hold only if a full sweep also happens every step.

This script replays every flight step by step, and each policy knows only
the beams it actually measured:

- full sweep every step (5 ms per step under the FR2 timing below);
- periodic: a full sweep every N steps, and in between a local search:
  the current beam, its +/-W index neighbours and its M strongest
  tie-graph partners (fitted on other flights only);
- every policy has a beam-loss fallback: if the best measured power drops
  more than DROP_DB below what the current beam delivered at the last
  step, a full sweep follows (one extra report round trip);
- genie: always the best beam, no training time (upper bound).

Metrics per step, against the full sweep the dataset recorded:
exact top-1, within-noise (0.25 dB), power ratio, beams measured, and
beam-training latency. Effective spectral efficiency over the interval
until the next step is

    (1 - latency / interval) * mean(log2(1 + SNR0 * p_used / p_best)

evaluated at both ends of the interval with the measured powers (so the
beam's ageing until the next decision is measured, not modelled). The
dataset's powers are unitless, so SNR0 (the best beam's SNR) is a nominal
operating point, reported at several values; it is not a property of
the testbed.

Splits: GroupKFold(5) over seq_index; the tie graph for a flight is fitted
on the other folds' flights.

Run:  python closed_loop_beam_tracking.py
Writes results/closed_loop_beam_tracking.json and .csv
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from beam_prediction import BEAMS, FOLDS, NOISE_TOLERANCE_DB, RESULTS_DIR, load_scenario23
from beam_training_overhead import BEAM_MEASUREMENT_MS, REPORT_ROUND_TRIP_MS, tie_graph

NOMINAL_SNR_DB = (0.0, 10.0, 20.0)
DROP_DB = 3.0


@dataclass(frozen=True)
class Policy:
    name: str
    full_every: int | None  # None: full sweep only at the start and on beam loss
    window: int = 0
    partners: int = 0


POLICIES = [
    Policy("full sweep every step", 1),
    Policy("window +/-1, full every 4", 4, window=1),
    Policy("window +/-2, full every 4", 4, window=2),
    Policy("window +/-1 + 2 tie partners, full every 4", 4, window=1, partners=2),
    Policy("window +/-1 + 4 tie partners, full every 4", 4, window=1, partners=4),
    Policy("window +/-1 + 4 tie partners, full every 8", 8, window=1, partners=4),
    Policy("window +/-1, loss-triggered sweeps only", None, window=1),
    Policy("window +/-1 + 4 tie partners, loss-triggered sweeps only", None, window=1, partners=4),
    Policy("window +/-1 + 6 tie partners, loss-triggered sweeps only", None, window=1, partners=6),
]


def run_flight(power: np.ndarray, policy: Policy, graph: np.ndarray) -> dict[str, np.ndarray]:
    """Replay one flight; return per-step chosen beam, beams measured, latency and the measured mask."""

    steps = len(power)
    chosen = np.empty(steps, dtype=int)
    measured_count = np.empty(steps, dtype=int)
    latency = np.empty(steps)
    full_sweeps = np.zeros(steps, dtype=bool)
    mask = np.zeros((steps, BEAMS), dtype=bool)
    current, reference = None, None

    for n in range(steps):
        periodic = policy.full_every is not None and n % policy.full_every == 0
        if current is None or periodic:
            current = int(power[n].argmax())
            measured_count[n], latency[n], full_sweeps[n] = BEAMS, BEAMS * BEAM_MEASUREMENT_MS, True
            mask[n] = True
        else:
            candidates = {current}
            candidates.update(b for b in range(current - policy.window, current + policy.window + 1)
                              if 0 <= b < BEAMS)
            candidates.update(int(b) for b in graph[current, :policy.partners])
            candidates = np.fromiter(candidates, dtype=int)
            best = int(candidates[power[n, candidates].argmax()])
            measured_count[n] = len(candidates)
            latency[n] = len(candidates) * BEAM_MEASUREMENT_MS
            mask[n, candidates] = True

            if 10.0 * np.log10(power[n, best] / reference) < -DROP_DB:
                best = int(power[n].argmax())
                latency[n] += REPORT_ROUND_TRIP_MS + (BEAMS - len(candidates)) * BEAM_MEASUREMENT_MS
                measured_count[n] = BEAMS
                full_sweeps[n] = True
                mask[n] = True
            current = best

        chosen[n] = current
        reference = power[n, current]

    return {"chosen": chosen, "beams": measured_count, "latency": latency, "full_sweep": full_sweeps,
            "mask": mask}


def effective_se(
    power: np.ndarray,
    chosen: np.ndarray,
    latency_ms: np.ndarray,
    interval_s: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Per-interval effective SE (bit/s/Hz), steps 0..n-2 of one flight."""

    snr = 10.0 ** (snr_db / 10.0)
    rows = np.arange(len(power) - 1)
    start = np.log2(1.0 + snr * power[rows, chosen[rows]] / power[rows].max(axis=1))
    end = np.log2(1.0 + snr * power[rows + 1, chosen[rows]] / power[rows + 1].max(axis=1))
    duty = np.clip(1.0 - latency_ms[rows] / (1e3 * interval_s), 0.0, 1.0)

    return duty * (start + end) / 2.0


def main() -> None:
    """Replay every flight under every policy and summarise."""

    power, _, sequences, time_s, _ = load_scenario23()
    folds = list(GroupKFold(n_splits=FOLDS).split(power, groups=sequences))
    flight_rows = []

    for train, test in folds:
        graph = tie_graph(power[train])
        for sequence in np.unique(sequences[test]):
            index = np.flatnonzero(sequences == sequence)
            index = index[np.argsort(time_s[index], kind="stable")]
            flight_power, flight_time = power[index], time_s[index]
            interval = np.diff(flight_time)
            if len(index) < 2 or np.any(interval <= 0):
                continue
            best = flight_power.argmax(axis=1)
            best_power = flight_power.max(axis=1)

            for policy in POLICIES + [Policy("genie (best beam, no training time)", 1)]:
                genie = policy.name.startswith("genie")
                trace = run_flight(flight_power, policy, graph)
                latency = np.zeros(len(index)) if genie else trace["latency"]
                rows = np.arange(len(index))
                loss_db = 10.0 * np.log10(best_power / flight_power[rows, trace["chosen"]])
                record = {
                    "policy": policy.name,
                    "sequence": int(sequence),
                    "steps": len(index),
                    "exact_top1": np.mean(trace["chosen"] == best),
                    "within_noise": np.mean(loss_db <= NOISE_TOLERANCE_DB),
                    "power_ratio": np.mean(flight_power[rows, trace["chosen"]] / best_power),
                    "beams_mean": trace["beams"].mean(),
                    "latency_ms_mean": latency.mean(),
                    "latency_ms_p95": np.percentile(latency, 95),
                    "full_sweep_fraction": trace["full_sweep"].mean(),
                }
                for snr_db in NOMINAL_SNR_DB:
                    se = effective_se(flight_power, trace["chosen"], latency, interval, snr_db)
                    record[f"se_{snr_db:g}dB_sum"] = se.sum()
                    record["intervals"] = len(se)
                flight_rows.append(record)

    flights = pd.DataFrame(flight_rows)

    # Pool over flights weighted by steps; SE as % of the genie's.
    def pooled(group: pd.DataFrame) -> pd.Series:
        weights = group["steps"]
        out = {column: np.average(group[column], weights=weights)
               for column in ("exact_top1", "within_noise", "power_ratio", "beams_mean",
                              "latency_ms_mean", "full_sweep_fraction")}
        out["latency_ms_p95_worst_flight"] = group["latency_ms_p95"].max()
        for snr_db in NOMINAL_SNR_DB:
            out[f"se_{snr_db:g}dB"] = group[f"se_{snr_db:g}dB_sum"].sum() / group["intervals"].sum()
        return pd.Series(out)

    summary = flights.groupby("policy", sort=False).apply(pooled)
    genie = summary.loc["genie (best beam, no training time)"]
    for snr_db in NOMINAL_SNR_DB:
        summary[f"se_{snr_db:g}dB_pct_of_genie"] = 100.0 * summary[f"se_{snr_db:g}dB"] / genie[f"se_{snr_db:g}dB"]

    RESULTS_DIR.mkdir(exist_ok=True)
    flights.to_csv(RESULTS_DIR / "closed_loop_beam_tracking_flights.csv", index=False)
    summary.round(5).to_csv(RESULTS_DIR / "closed_loop_beam_tracking.csv")
    (RESULTS_DIR / "closed_loop_beam_tracking.json").write_text(json.dumps({
        "split": f"GroupKFold({FOLDS}) over seq_index; tie graph fitted on other folds",
        "beam_measurement_ms": BEAM_MEASUREMENT_MS,
        "report_round_trip_ms": REPORT_ROUND_TRIP_MS,
        "drop_db": DROP_DB,
        "nominal_snr_db": list(NOMINAL_SNR_DB),
        "median_step_interval_s": float(np.median(np.diff(time_s)[np.diff(sequences) == 0])),
        "summary": summary.round(5).reset_index().to_dict(orient="records"),
    }, indent=2))

    pd.set_option("display.width", 250)
    columns = ["exact_top1", "within_noise", "power_ratio", "beams_mean", "latency_ms_mean",
               "full_sweep_fraction"] + [f"se_{s:g}dB_pct_of_genie" for s in NOMINAL_SNR_DB]
    print(summary[columns].round(3).to_string())


if __name__ == "__main__":
    main()
