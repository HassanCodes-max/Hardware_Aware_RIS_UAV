"""Regenerate every paper figure from the JSON/CSV files in results/.

Run after the experiment scripts:  python make_figures.py
Writes paper/figures/*.png (300 dpi, single IEEE column width).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"

# Validated categorical slots 1-3 (pass all-pairs CVD checks), plus a
# neutral grey for baselines; ink colours for text.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#8c8b86"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

COLUMN_WIDTH_IN = 3.5

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name)
    plt.close(figure)
    print(f"wrote paper/figures/{name}")


def target_line(axis, value: float) -> None:
    axis.axhline(value, color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=1)
    axis.text(axis.get_xlim()[1], value, " 85%", va="center", ha="left", color=INK_MUTED, fontsize=7)


def figure_beam_training() -> None:
    """Exact top-1 and macro-F1 vs beams measured at t+1 (two panels, one axis each)."""

    curves = pd.read_csv(RESULTS / "beam_training_overhead_curves.csv")
    curves = curves[(curves["horizon"] == 1) & curves["shortlist"].str.startswith("K=")]
    ceiling = json.loads((RESULTS / "beam_training_overhead.json").read_text())["noncausal_oracle_ceiling"]

    series = [
        ("gru_soft+sweep", "GRU + last sweep (AI)", BLUE, "o"),
        ("window", "Window around last beam", ORANGE, "s"),
        ("last_sweep", "Last-sweep ranking", AQUA, "^"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.3))
    for axis, metric, label, oracle in (
        (axes[0], "exact_top1", "Exact top-1 accuracy", ceiling["exact_top1"]),
        (axes[1], "macro_f1", "Macro-F1 (56 classes)", ceiling["macro_f1"]),
    ):
        for method, name, color, marker in series:
            rows = curves[curves["method"] == method].sort_values("mean_beams_measured")
            axis.plot(rows["mean_beams_measured"], rows[metric], color=color, linewidth=1.5,
                      marker=marker, markersize=4, label=name)
        axis.set_xlim(0.5, 16.5)
        axis.set_xticks([1, 2, 4, 6, 8, 12, 16])
        axis.set_ylim(0.3, 1.01)
        axis.set_xlabel("Beams measured per decision (of 64)")
        axis.set_ylabel(label)
        target_line(axis, 0.85)
        axis.axhline(oracle, color=GREY, linewidth=0.8, linestyle=":")
        axis.text(16.3, oracle - 0.04, "blind-prediction ceiling", ha="right", color=INK_MUTED, fontsize=7)

    axes[0].legend(loc="lower right", fontsize=7)
    figure.tight_layout()
    save(figure, "fig_beam_training.png")


def figure_tracking() -> None:
    """Delivered array gain vs prediction horizon for each tracker."""

    summary = pd.DataFrame(json.loads((RESULTS / "tracking_metrics.json").read_text())["summary"])
    series = [
        ("kalman_cv", "Kalman CV", BLUE, "o"),
        ("constant_velocity", "Constant velocity", ORANGE, "s"),
        ("persistence", "Persistence", GREY, "^"),
    ]

    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.2))
    for method, name, color, marker in series:
        rows = summary[summary["method"] == method].sort_values("horizon_ms")
        axis.plot(rows["horizon_ms"], rows["mean_array_gain_pct"], color=color, linewidth=1.5,
                  marker=marker, markersize=4, label=name)
    axis.set_xscale("log")
    axis.set_xticks([10, 100, 500], ["10", "100", "500"])
    axis.set_xlabel("Prediction horizon (ms)")
    axis.set_ylabel("Array gain delivered (%)")
    axis.set_ylim(50, 101)
    axis.legend(loc="lower left", fontsize=7)
    save(figure, "fig_tracking_gain.png")


def figure_impairments() -> None:
    """SNDR penalty of each hardware impairment alone (single series)."""

    ablation = json.loads((RESULTS / "physical_layer_metrics.json").read_text())["impairment_ablation"]
    labels = {
        "ris_amplitude_phase_coupling": "RIS amplitude-phase coupling",
        "pa_nonlinearity": "PA nonlinearity (6 dB IBO)",
        "s11_mismatch": "S11 mismatch (-10 dB)",
        "ris_phase_quantization": "2-bit phase quantization",
        "ris_phase_noise": "RIS phase noise (5 deg)",
        "all_combined": "All combined",
    }
    items = sorted(
        ((labels[key], value) for key, value in ablation["penalty_db"].items()),
        key=lambda item: item[1],
    )

    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.0))
    names, values = zip(*items)
    bars = axis.barh(names, values, color=[BLUE if n != "All combined" else ORANGE for n in names],
                     height=0.6)
    for bar, value in zip(bars, values):
        axis.text(value + 0.1, bar.get_y() + bar.get_height() / 2, f"{value:.2f} dB",
                  va="center", color=INK, fontsize=7)
    axis.set_xlabel("SNDR penalty vs ideal hardware (dB)")
    axis.set_xlim(0, max(values) * 1.25)
    axis.grid(axis="y", visible=False)
    save(figure, "fig_impairment_ablation.png")


def figure_ris_sizing() -> None:
    """Link availability vs RIS size, ideal vs realistic hardware."""

    sizing = json.loads((RESULTS / "physical_layer_metrics.json").read_text())["ris_sizing_availability"]
    sizes = list(sizing)
    x = np.arange(len(sizes))

    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.1))
    width = 0.36
    for offset, key, name, color in (
        (-width / 2, "ideal_availability_pct", "Ideal hardware", GREY),
        (width / 2, "hardware_aware_availability_pct", "Realistic hardware, aware control", BLUE),
    ):
        values = [sizing[s][key] for s in sizes]
        bars = axis.bar(x + offset, values, width=width - 0.03, color=color, label=name)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}",
                      ha="center", color=INK, fontsize=6.5)
    axis.set_xticks(x, [f"{s}\n({sizing[s]['aperture_cm']:.0f} cm)" for s in sizes])
    axis.set_ylabel("Availability, SNDR >= 0 dB (%)")
    axis.set_ylim(0, 110)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    save(figure, "fig_ris_sizing.png")


def main() -> None:
    figure_beam_training()
    figure_tracking()
    figure_impairments()
    figure_ris_sizing()


if __name__ == "__main__":
    main()
