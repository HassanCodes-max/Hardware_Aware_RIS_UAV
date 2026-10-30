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
results/gps_camera/, whichever exist), and the nested stacks of blind_stacking.py
(results/stacked_power/, results/stacked_all/) against the GRU ensemble and each other.
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

SIDE_INFORMATION = {"power only": ROOT, "+ GPS": ROOT / "gps", "+ camera emb.": ROOT / "camera",
                    "+ GPS + camera emb.": ROOT / "gps_camera", "+ UAV pixel": ROOT / "uav_pixel",
                    "+ GPS + UAV pixel": ROOT / "gps_uav_pixel"}
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


def blind_per_flight(scores: np.ndarray, data) -> pd.DataFrame:
    """Per-flight within-noise and exact top-1 of a (samples, 64) blind score matrix."""

    chosen = scores.argmax(axis=1)
    power = data["target_power"]
    loss_db = 10.0 * np.log10(power.max(axis=1) / power[np.arange(len(chosen)), chosen])
    frame = pd.DataFrame({"flight": data["groups"],
                          "within_noise": loss_db <= NOISE_TOLERANCE_DB,
                          "exact_top1": chosen == data["target"]})
    return frame.groupby("flight").agg(within_noise=("within_noise", "mean"),
                                       exact_top1=("exact_top1", "mean"),
                                       samples=("within_noise", "size"))


def compare(family: str, a_name: str, a: pd.DataFrame, b_name: str, b: pd.DataFrame, rng) -> list[dict]:
    a = a.loc[b.index]
    return [{"family": family, "metric": label, "a": a_name, "b": b_name,
             **paired_test(100.0 * a[metric].to_numpy(), 100.0 * b[metric].to_numpy(),
                           b["samples"].to_numpy(float), rng)}
            for metric, label in (("within_noise", "within noise"), ("exact_top1", "exact top-1"))]


def stacking_family(rng) -> list[dict]:
    """Nested stacks of blind_stacking.py against the GRU ensemble and against each other."""

    rows = []
    for horizon in (1, 3):
        paths = {name: ROOT / folder / f"beam_sequence_oof_h{horizon}.npz"
                 for name, folder in (("power only", "stacked_power"), ("all sensing", "stacked_all"))}
        if not all(p.exists() for p in paths.values()):
            continue
        power, sensing = np.load(paths["power only"]), np.load(paths["all sensing"])
        gru = blind_per_flight(power[BLIND_MODEL], power)
        stack_power = blind_per_flight(power["stacked"], power)
        stack_all = blind_per_flight(sensing["stacked"], sensing)
        family = f"stacking t+{horizon}"
        rows += compare(family, "Stacked, power only", stack_power, "GRU ensemble, power only", gru, rng)
        rows += compare(family, "Stacked, all sensing", stack_all, "GRU ensemble, power only", gru, rng)
        rows += compare(family, "Stacked, all sensing", stack_all, "Stacked, power only", stack_power, rng)
    return rows


def blind_family(rng) -> list[dict]:
    per_flight = {}
    for name, folder in SIDE_INFORMATION.items():
        for horizon in (1, 3):
            path = folder / f"beam_sequence_oof_h{horizon}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            per_flight[(name, horizon)] = blind_per_flight(data[BLIND_MODEL], data)

    rows = []
    for horizon in (1, 3):
        base = per_flight.get(("power only", horizon))
        for name in list(SIDE_INFORMATION)[1:]:
            other = per_flight.get((name, horizon))
            if base is None or other is None:
                continue
            rows += compare(f"blind t+{horizon}", f"GRU ensemble, {name}", other,
                            "GRU ensemble, power only", base, rng)
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = closed_loop_family(rng) + blind_family(rng) + stacking_family(rng)
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
