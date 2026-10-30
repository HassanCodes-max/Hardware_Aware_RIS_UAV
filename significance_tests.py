"""
Paired per-flight significance tests between the methods compared in the paper.

The unit of replication is the flight (seq_index), never the sample:
consecutive samples of one flight are strongly correlated, so a
sample-level test would overstate significance. For every comparison A
vs B on the same held-out flights we report

* the pooled difference (A - B, weighted by samples, as in the tables);
* its 95% CI from a paired bootstrap over flights (5000 resamples);
* the Wilcoxon signed-rank p-value on per-flight differences (Pratt
  handling of zero differences), Holm-corrected within each family;
* how many flights A wins, ties and loses.

Families: closed-loop tracking policies (results/closed_loop_beam_tracking_flights.csv),
and blind prediction with and without side information (out-of-fold
predictions of the GRU ensemble in results/, results/gps/, results/camera/,
results/gps_camera/, whichever exist).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from beam_prediction import NOISE_TOLERANCE_DB

ROOT = Path(__file__).resolve().parent / "results"
RESAMPLES = 5000
SEED = 7

TRACKER = "window +/-1 + 4 tie partners, full every 8"
CLOSED_LOOP = [
    # (A, B): is A better than B?
    (TRACKER, "full sweep every step"),
    ("window +/-1 + 6 tie partners, full every 8", TRACKER),
    ("window +/-1 + 8 tie partners, full every 8", "window +/-1 + 6 tie partners, full every 8"),
    ("window +/-1 + 4 tie partners, full every 4", TRACKER),
    (TRACKER, "window +/-1, full every 4"),
    ("window +/-1 + 6 tie partners, loss-triggered sweeps only", TRACKER),
]
CLOSED_LOOP_METRICS = {"exact_top1": "exact top-1", "se_10dB": "effective SE, 10 dB"}

SIDE_INFORMATION = {"power only": ROOT, "+ GPS": ROOT / "gps", "+ camera": ROOT / "camera",
                    "+ GPS + camera": ROOT / "gps_camera"}
BLIND_MODEL = "gru_soft_aug_ens"


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


def paired_test(a: np.ndarray, b: np.ndarray, weights: np.ndarray, rng) -> dict:
    """a, b: per-flight values of the two methods on the same flights."""

    difference = a - b
    pooled = float(np.average(difference, weights=weights))
    draws = rng.integers(0, len(a), size=(RESAMPLES, len(a)))
    boot = (difference[draws] * weights[draws]).sum(axis=1) / weights[draws].sum(axis=1)
    if np.allclose(difference, 0):
        p = 1.0
    else:
        p = float(wilcoxon(a, b, zero_method="pratt").pvalue)
    return {"difference": pooled,
            "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
            "wilcoxon_p": p, "flights": int(len(a)),
            "wins": int((difference > 1e-12).sum()), "ties": int((np.abs(difference) <= 1e-12).sum()),
            "losses": int((difference < -1e-12).sum())}


def closed_loop_family(rng) -> list[dict]:
    flights = pd.read_csv(ROOT / "closed_loop_beam_tracking_flights.csv")
    flights["se_10dB"] = flights["se_10dB_sum"] / flights["intervals"]
    genie = flights[flights["policy"].str.startswith("genie")].set_index("sequence")["se_10dB"]
    flights["se_10dB"] = 100.0 * flights["se_10dB"] / flights["sequence"].map(genie)

    rows = []
    for metric, label in CLOSED_LOOP_METRICS.items():
        for a_name, b_name in CLOSED_LOOP:
            a = flights[flights["policy"] == a_name].set_index("sequence")
            b = flights[flights["policy"] == b_name].set_index("sequence").loc[a.index]
            rows.append({"family": "closed loop", "metric": label, "a": a_name, "b": b_name,
                         **paired_test(a[metric].to_numpy(), b[metric].to_numpy(),
                                       a["steps"].to_numpy(float), rng)})
    return rows


def blind_family(rng) -> list[dict]:
    per_flight = {}
    for name, folder in SIDE_INFORMATION.items():
        for horizon in (1, 3):
            path = folder / f"beam_sequence_oof_h{horizon}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            chosen = data[BLIND_MODEL].argmax(axis=1)
            power = data["target_power"]
            loss_db = 10.0 * np.log10(power.max(axis=1) / power[np.arange(len(chosen)), chosen])
            frame = pd.DataFrame({"flight": data["groups"],
                                  "within_noise": loss_db <= NOISE_TOLERANCE_DB,
                                  "exact_top1": chosen == data["target"]})
            per_flight[(name, horizon)] = frame.groupby("flight").agg(
                within_noise=("within_noise", "mean"), exact_top1=("exact_top1", "mean"),
                samples=("within_noise", "size"))

    rows = []
    for horizon in (1, 3):
        base = per_flight.get(("power only", horizon))
        for name in list(SIDE_INFORMATION)[1:]:
            other = per_flight.get((name, horizon))
            if base is None or other is None:
                continue
            other = other.loc[base.index]
            for metric, label in (("within_noise", "within noise"), ("exact_top1", "exact top-1")):
                rows.append({"family": f"blind t+{horizon}", "metric": label,
                             "a": f"GRU ensemble, {name}", "b": "GRU ensemble, power only",
                             **paired_test(100.0 * other[metric].to_numpy(),
                                           100.0 * base[metric].to_numpy(),
                                           base["samples"].to_numpy(float), rng)})
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = closed_loop_family(rng) + blind_family(rng)
    table = pd.DataFrame(rows)
    for metric_family, group in table.groupby(["family", "metric"]).groups.items():
        table.loc[group, "holm_p"] = holm(table.loc[group, "wilcoxon_p"].tolist())
    for column in ("difference", "ci95_low", "ci95_high"):
        scale = np.where(table["metric"].isin(["exact top-1"]) & (table["family"] == "closed loop"),
                         100.0, 1.0)
        table[column] = table[column] * scale   # report every difference in points

    table.to_csv(ROOT / "significance_tests.csv", index=False)
    (ROOT / "significance_tests.json").write_text(json.dumps({
        "unit": "flight (seq_index); differences in percentage points",
        "bootstrap_resamples": RESAMPLES,
        "test": "Wilcoxon signed-rank (Pratt), Holm within family and metric",
        "rows": table.round(6).to_dict(orient="records"),
    }, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(table[["family", "metric", "a", "b", "difference", "ci95_low", "ci95_high",
                 "wilcoxon_p", "holm_p", "wins", "ties", "losses"]].round(4).to_string())


if __name__ == "__main__":
    main()
