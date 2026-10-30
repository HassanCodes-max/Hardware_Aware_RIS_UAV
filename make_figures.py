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
YELLOW = "#eda100"   # slot 4: only with a distinct marker shape
MAGENTA = "#e87ba4"  # slot 5: likewise
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
    axis.text(axis.get_xlim()[1], value, f" {value:.0%}", va="center", ha="left", color=INK_MUTED, fontsize=7)


def figure_beam_training() -> None:
    """Exact top-1 and macro-F1 vs beams measured, t+1 (top) and t+3 (bottom)."""

    curves = pd.read_csv(RESULTS / "beam_training_overhead_curves.csv")
    ceiling = json.loads((RESULTS / "beam_training_overhead.json").read_text())["noncausal_oracle_ceiling"]

    # Burst rows are labelled "K=first+partners"; show the 4-partner series.
    series = [
        ("gru_aug_ens burst", r"K=\d+\+4$", "GRU ensemble + tie graph, burst (AI)", BLUE, "o"),
        ("gru_aug_ens+sweep", r"K=\d+$", "GRU ensemble + last sweep (AI)", AQUA, "D"),
        ("window", r"K=\d+$", "Window around last beam", ORANGE, "s"),
        ("last_sweep", r"K=\d+$", "Last-sweep ranking", GREY, "^"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH_IN * 2, 4.4), sharex=True)
    for row, horizon in enumerate((1, 3)):
        at_horizon = curves[curves["horizon"] == horizon]
        for axis, metric, label, oracle in (
            (axes[row, 0], "exact_top1", "Exact top-1", ceiling["exact_top1"]),
            (axes[row, 1], "macro_f1", "Macro-F1 (56 classes)", ceiling["macro_f1"]),
        ):
            for method, pattern, name, color, marker in series:
                rows = at_horizon[(at_horizon["method"] == method)
                                  & at_horizon["shortlist"].str.fullmatch(pattern)]
                rows = rows.sort_values("mean_beams_measured")
                axis.plot(rows["mean_beams_measured"], rows[metric], color=color, linewidth=1.5,
                          marker=marker, markersize=4, label=name)
            axis.set_xlim(0.5, 16.5)
            axis.set_xticks([1, 2, 4, 6, 8, 12, 16])
            axis.set_ylim(0.3, 1.01)
            axis.set_ylabel(f"{label}, t+{horizon}")
            target_line(axis, 0.95)
            axis.axhline(oracle, color=GREY, linewidth=0.8, linestyle=":")
            axis.text(16.3, oracle - 0.05, "blind-prediction ceiling", ha="right", color=INK_MUTED, fontsize=7)

    for axis in axes[1]:
        axis.set_xlabel("Beams measured per decision (of 64)")
    axes[0, 0].legend(loc="lower right", fontsize=6.5)
    figure.tight_layout()
    save(figure, "fig_beam_training.png")


def figure_tracking() -> None:
    """Delivered array gain vs prediction horizon for each tracker."""

    summary = pd.DataFrame(json.loads((RESULTS / "tracking_metrics.json").read_text())["summary"])
    series = [
        ("kalman_cv", "Kalman CV", BLUE, "o"),
        ("learned_ridge", "Learned ridge", AQUA, "D"),
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
    """Availability vs RIS size: 60 GHz ideal vs realistic (left), 140 GHz vs receiver NF (right)."""

    metrics = json.loads((RESULTS / "physical_layer_metrics.json").read_text())
    sizing = metrics["ris_sizing_availability"]
    sizes = list(sizing)
    x = np.arange(len(sizes))

    figure, (left, right) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.3),
                                         gridspec_kw={"width_ratios": [1.1, 1]})
    width = 0.36
    for offset, key, name, color in (
        (-width / 2, "ideal_availability_pct", "Ideal hardware", GREY),
        (width / 2, "hardware_aware_availability_pct", "Realistic hardware, aware control", BLUE),
    ):
        values = [sizing[s][key] for s in sizes]
        bars = left.bar(x + offset, values, width=width - 0.03, color=color, label=name)
        for bar, value in zip(bars, values):
            left.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}",
                      ha="center", color=INK, fontsize=6.5)
    left.set_xticks(x, [f"{s}\n({sizing[s]['aperture_cm']:.0f} cm)" for s in sizes])
    left.set_ylabel("Availability, SNDR >= 0 dB (%)")
    left.set_ylim(0, 140); left.set_yticks([0, 20, 40, 60, 80, 100])
    left.set_title("60 GHz", fontsize=8, color=INK)
    left.grid(axis="x", visible=False)
    left.legend(loc="upper left", fontsize=6.5)

    # 140 GHz with D-band hardware: realistic availability per receiver noise figure.
    nf10 = dict(metrics["ris_sizing_140ghz_dband_hardware"])
    nf10.update(metrics["ris_sizing_140ghz_nf_sensitivity"]["nf_10db"])
    curves = [("NF 10 dB (default)", nf10, BLUE, "o"),
              ("NF 7 dB", metrics["ris_sizing_140ghz_nf_sensitivity"]["nf_7db"], ORANGE, "s"),
              ("NF 6 dB", metrics["ris_sizing_140ghz_nf_sensitivity"]["nf_6db"], AQUA, "^")]
    for name, table, color, marker in curves:
        rows = sorted(table.values(), key=lambda r: r["aperture_cm"])
        right.plot([r["aperture_cm"] for r in rows], [r["hardware_aware_availability_pct"] for r in rows],
                   color=color, marker=marker, markersize=4, linewidth=1.5, label=name)
    ideal = sorted(nf10.values(), key=lambda r: r["aperture_cm"])
    right.plot([r["aperture_cm"] for r in ideal], [r["ideal_availability_pct"] for r in ideal],
               color=GREY, linestyle="--", linewidth=1.2, label="Ideal hardware, NF 10 dB")
    right.set_xlabel("RIS aperture (cm), 140 GHz")
    right.set_ylabel("Availability, realistic RIS (%)")
    right.set_ylim(0, 105)
    right.set_xticks([8, 16, 20, 24, 28, 32])
    right.set_title("140 GHz, D-band transceiver", fontsize=8, color=INK)
    right.legend(loc="lower right", fontsize=6.5)
    figure.tight_layout()
    save(figure, "fig_ris_sizing.png")


def figure_system() -> None:
    """Block diagram: hardware-aware link (top) and the sensing-aided beam-management loop (bottom)."""

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN * 2, 2.9))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 60)
    axis.axis("off")

    def box(x, y, w, h, title, body, edge=INK_MUTED, fill="#f4f3ef"):
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                      linewidth=0.9, edgecolor=edge, facecolor=fill))
        axis.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.2, color=INK,
                  fontweight="bold")
        axis.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=6.2, color=INK_MUTED,
                  linespacing=1.25)

    def arrow(x0, y0, x1, y1, text=None, dx=0.0, dy=1.4, style="-|>", color=INK_MUTED):
        axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
                                       linewidth=0.9, color=color))
        if text:
            axis.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, ha="center", va="bottom",
                      fontsize=6, color=INK_MUTED)

    # Row 1: the physical link (Secs. II-IV).
    axis.text(1, 58.5, "Hardware-aware link (simulated, Secs. II–IV)", fontsize=7.4, color=BLUE,
              fontweight="bold", va="top")
    box(1, 34, 20, 19, "Base station", "PA (Saleh, Bussgang)\n64-beam codebook\nS11 mismatch")
    box(33, 34, 23, 19, "RIS", "2-bit phases, phase noise\namplitude–phase coupling\ncoupling-aware control")
    box(65, 34, 16, 19, "UAV", "300 m flight\n60 / 140 GHz")
    box(86, 34, 13, 19, "Channel", "LS / LMMSE /\nparametric\nestimation")
    arrow(21.8, 43.5, 32.2, 43.5, "gases,\nITU-R P.676", dy=1.0)
    arrow(56.6, 43.5, 64.4, 43.5)
    arrow(81.6, 43.5, 85.4, 43.5)

    # Row 2: beam management on real measurements (Secs. V-VI).
    axis.text(1, 28.5, "Beam management on DeepSense Scenario 23 (measured, Secs. V–VI)", fontsize=7.4,
              color=ORANGE, fontweight="bold", va="top")
    box(1, 2, 19, 21, "Sensing (t − 1)", "BS camera: UAV pixel\n(label-free locator)\nUAV GPS")
    box(25, 2, 23, 21, "Candidate beams", "current beam ± 1\ntie-graph partners\ntop-k of camera/GPS RF")
    box(53, 2, 20, 21, "Measure", "K SSB beams\n78 µs each\n(5G NR FR2)")
    box(78, 2, 21, 21, "Decide", "keep strongest beam\nloss > 3 dB:\nfull 64-beam sweep")
    arrow(20.6, 12.5, 24.4, 12.5)
    arrow(48.6, 12.5, 52.4, 12.5)
    arrow(73.6, 12.5, 77.4, 12.5)
    axis.add_patch(FancyArrowPatch((88.5, 1.2), (36.5, 1.2), connectionstyle="arc3,rad=-0.12",
                                   arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=INK_MUTED))
    axis.text(62, -4.6, "next step: chosen beam becomes the current beam", ha="center", va="top",
              fontsize=6, color=INK_MUTED)
    axis.set_ylim(-8, 60)
    save(figure, "fig_system.png")


def figure_conformal() -> None:
    """Achieved exact top-1 (= set coverage) and set size vs the conformal target."""

    table = pd.read_csv(RESULTS / "conformal_beam_training.csv")
    table = table[table["method"] == "gru_aug_ens rank"]

    figure, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.3))
    for horizon, color, marker in ((1, BLUE, "o"), (3, ORANGE, "s")):
        rows = table[table["horizon"] == horizon].sort_values("target_coverage")
        target = 100 * rows["target_coverage"]
        achieved, worst = 100 * rows["exact_top1"], 100 * rows["min_fold_coverage"]
        axes[0].plot(target, achieved, color=color, linewidth=1.5, marker=marker, markersize=4,
                     label=f"t+{horizon}, pooled")
        axes[0].plot(target, worst, color=color, linewidth=1.0, linestyle=":", marker=marker,
                     markersize=3, markerfacecolor="white", label=f"t+{horizon}, worst fold")
        axes[1].plot(target, rows["mean_beams_measured"], color=color, linewidth=1.5, marker=marker,
                     markersize=4, label=f"t+{horizon}")

    axes[0].plot([89, 100], [89, 100], color=GREY, linewidth=0.8, linestyle="--", zorder=1)
    axes[0].set_xlim(89, 100)
    axes[0].set_ylim(88, 100)
    axes[0].set_xlabel("Target exact top-1 (%)")
    axes[0].set_ylabel("Achieved exact top-1 (%)")
    axes[0].legend(loc="lower right", fontsize=6.5)
    axes[1].set_xlim(89, 100)
    axes[1].set_xlabel("Target exact top-1 (%)")
    axes[1].set_ylabel("Beams measured (of 64)")
    axes[1].legend(loc="upper left", fontsize=7)
    figure.tight_layout()
    save(figure, "fig_conformal.png")


def figure_closed_loop() -> None:
    """Closed loop: all policies (left) and the equal-budget learned-vs-graph comparison (right)."""

    trackers = pd.DataFrame(json.loads((RESULTS / "closed_loop_beam_tracking.json").read_text())["summary"])
    trackers = trackers[~trackers["policy"].str.startswith("genie")]
    learned = pd.read_csv(RESULTS / "closed_loop_learned_tracking.csv")
    partners = learned[learned["config"].str.startswith("tracker")].sort_values("beams_mean")
    models = learned[~learned["config"].str.startswith("tracker")]

    figure, (overview, zoom) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.5))

    full = trackers["policy"].str.startswith("full sweep")
    overview.scatter(trackers.loc[full, "latency_ms_mean"], 100 * trackers.loc[full, "exact_top1"],
                     color=GREY, marker="D", s=22, zorder=3, label="Full sweep every step")
    overview.scatter(trackers.loc[~full, "latency_ms_mean"], 100 * trackers.loc[~full, "exact_top1"],
                     color=BLUE, marker="o", s=18, zorder=3, label="Window / tie-graph trackers")
    overview.scatter(partners["latency_ms_mean"], 100 * partners["exact_top1"], color=BLUE, marker="o",
                     s=18, zorder=3)
    overview.scatter(models["latency_ms_mean"], 100 * models["exact_top1"], color=ORANGE, marker="s",
                     s=12, zorder=2, label="Tracker + learned beams")
    overview.set_xscale("log")
    overview.set_xticks([0.3, 0.5, 1, 2, 5])
    overview.set_xticklabels(["0.3", "0.5", "1", "2", "5"])
    overview.set_ylim(70, 101)
    overview.set_xlabel("Mean beam-training latency per decision (ms)")
    overview.set_ylabel("Exact top-1, closed loop (%)")
    target_line(overview, 95)
    overview.texts[-1].set_text(" 95%")
    overview.legend(loc="lower right", fontsize=6.5)

    # Equal budget: x = beams measured; extra beams are either tie partners or model picks.
    zoom.plot(partners["beams_mean"], 100 * partners["exact_top1"], color=BLUE, linewidth=1.5, marker="o",
              markersize=5, zorder=3, label="Tracker, 4 / 6 / 8 tie partners")
    for extra in sorted(models["extra_beams"].unique()):
        group = models[models["extra_beams"] == extra].reset_index(drop=True)
        offsets = np.linspace(-0.12, 0.12, len(group))
        zoom.scatter(group["beams_mean"] + offsets, 100 * group["exact_top1"], color=ORANGE, marker="s", s=14,
                     zorder=2, label="Tracker + learned beams (10 models)" if extra == 2 else None)
    zoom.set_xticks(partners["beams_mean"].round(1))
    zoom.set_xticklabels([f"{b:.1f}\n({lat:.2f} ms)" for b, lat in
                          zip(partners["beams_mean"], partners["latency_ms_mean"])])
    zoom.set_xlabel("Beams measured per decision (latency)")
    zoom.set_ylabel("Exact top-1, closed loop (%)")
    zoom.set_ylim(96.5, 99.8)
    zoom.legend(loc="lower right", fontsize=6.5)

    figure.tight_layout()
    save(figure, "fig_closed_loop.png")


GPS_MODELS = [
    ("beam_sequence_metrics.csv", "persistence", "Persistence"),
    ("beam_prediction_metrics.csv", "random_forest+sweep_prior", "RF + sweep prior"),
    ("beam_sequence_metrics.csv", "gru_soft_target", "GRU, soft target"),
    ("beam_sequence_metrics.csv", "gru_soft_aug_ens", "GRU ensemble"),
]


SIDE_INFORMATION = [
    # (label, results folder, colour, marker); yellow is CVD-close to orange, so every
    # series also has its own marker shape.
    ("Power only", "power_only", BLUE, "o"),
    ("+ GPS", "gps", ORANGE, "s"),
    ("+ camera emb.", "camera", AQUA, "^"),
    ("+ GPS + camera emb.", "gps_camera", YELLOW, "D"),
    ("+ UAV pixel", "uav_pixel", MAGENTA, "v"),
]


def side_information_comparison() -> pd.DataFrame:
    """Blind-prediction metrics with each set of side information (missing runs skipped)."""

    rows = []
    for features, folder, _, _ in SIDE_INFORMATION:
        for file, model, name in GPS_MODELS:
            path = RESULTS / folder / file
            if not path.exists():
                continue
            table = pd.read_csv(path)
            for _, row in table[table["model"] == model].iterrows():
                rows.append({"features": features, "model": name, "horizon": int(row["horizon"]),
                             **{c: row[c] for c in ("within_noise_acc", "balanced_within_noise_acc",
                                                    "top1_acc", "macro_f1", "power_ratio")}})
    return pd.DataFrame(rows)


def figure_side_information() -> None:
    """Within-noise accuracy per model and horizon, for every set of side information."""

    table = side_information_comparison()
    names = [name for _, _, name in GPS_MODELS if name != "Persistence"]
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.1))
    for axis, horizon, xlim in zip(axes, (1, 3), ((92, 97.5), (73, 89))):
        rows = table[(table["horizon"] == horizon) & table["model"].isin(names)]
        y = np.arange(len(names))[::-1]
        values = rows.pivot(index="model", columns="features", values="within_noise_acc").loc[names] * 100
        axis.hlines(y, values.min(axis=1), values.max(axis=1), color=GRID, linewidth=2.5, zorder=1)
        offsets = np.linspace(0.18, -0.18, len(SIDE_INFORMATION))
        for (features, _, colour, marker), offset in zip(SIDE_INFORMATION, offsets):
            if features in values:
                axis.scatter(values[features], y + offset, color=colour, marker=marker, s=22,
                             zorder=3, label=features, edgecolors="white", linewidths=0.4)
        base = table[(table["horizon"] == horizon) & (table["model"] == "Persistence")
                     & (table["features"] == "Power only")]["within_noise_acc"]
        if len(base):
            axis.axvline(100 * base.iloc[0], color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=1)
            axis.text(100 * base.iloc[0], len(names) - 0.45, " persistence", fontsize=6.5,
                      color=INK_MUTED, va="bottom")
        axis.set_yticks(y)
        axis.set_yticklabels(names if horizon == 1 else [""] * len(names))
        axis.set_ylim(-0.6, len(names) - 0.1)
        axis.set_title(f"t+{horizon}", fontsize=8, color=INK)
        axis.set_xlim(*xlim)
        axis.set_xlabel("Within noise, blind prediction (%)")
        axis.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=7,
                  bbox_to_anchor=(0.5, 1.06))
    figure.tight_layout()
    save(figure, "fig_side_information.png")


SIGNIFICANCE_LABELS = {
    "window +/-1 + 4 tie partners, full every 8": "4 partners /8",
    "window +/-1 + 6 tie partners, full every 8": "6 partners /8",
    "window +/-1 + 8 tie partners, full every 8": "8 partners /8",
    "window +/-1 + 4 tie partners, full every 4": "4 partners /4",
    "window +/-1, full every 4": "window only /4",
    "window +/-1 + 6 tie partners, loss-triggered sweeps only": "6 partners, loss-trig.",
    "full sweep every step": "full sweep",
}


def figure_significance() -> None:
    """Paired per-flight differences with flight-bootstrap 95% CIs."""

    table = pd.read_csv(RESULTS / "significance_tests.csv")
    panels = [
        ("Closed loop: exact top-1", (table["family"] == "closed loop") & (table["metric"] == "exact top-1"),
         lambda r: f"{SIGNIFICANCE_LABELS[r['a']]} vs {SIGNIFICANCE_LABELS[r['b']]}"),
        ("Blind prediction: within noise", table["family"].str.startswith("blind") & (table["metric"] == "within noise"),
         lambda r: f"{r['family'].split()[1]}: {r['a'].split(', ', 1)[1]}"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.4))
    for axis, (title, mask, label) in zip(axes, panels):
        rows = table[mask].reset_index(drop=True)
        y = np.arange(len(rows))[::-1]
        significant = rows["holm_p"] < 0.05
        axis.axvline(0, color=INK_MUTED, linewidth=0.8, zorder=1)
        axis.hlines(y, rows["ci95_low"], rows["ci95_high"], color=BLUE, linewidth=1.6, zorder=2)
        axis.scatter(rows["difference"][significant], y[significant], color=BLUE, s=24, zorder=3,
                     label="Holm p < 0.05")
        axis.scatter(rows["difference"][~significant], y[~significant], facecolor="white",
                     edgecolor=BLUE, s=24, zorder=3, linewidths=1.2, label="not significant")
        axis.set_yticks(y)
        axis.set_yticklabels([label(r) for _, r in rows.iterrows()], fontsize=7)
        axis.set_title(title, fontsize=8, color=INK)
        axis.set_xlabel("Difference, A - B (points)")
        axis.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.07))
    figure.tight_layout()
    save(figure, "fig_significance.png")


def figure_side_tracking() -> None:
    """Closed loop with camera / GPS candidates: accuracy and throughput against latency."""

    summary = pd.DataFrame(json.loads((RESULTS / "closed_loop_side_tracking.json").read_text())["summary"])
    summary = summary[~summary["policy"].str.startswith("genie")]
    groups = [
        ("No side information", lambda p: not any(k in p for k in ("camera", "GPS")), GREY, "o"),
        ("Camera (UAV pixel)", lambda p: p.startswith("camera top"), AQUA, "^"),
        ("GPS", lambda p: p.startswith("GPS"), ORANGE, "s"),
        ("Camera + GPS", lambda p: p.startswith("camera + GPS"), BLUE, "D"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN * 2, 2.4))
    for axis, column, scale, label, ylim in (
            (axes[0], "exact_top1", 100.0, "Exact top-1, closed loop (%)", (91.5, 100.5)),
            (axes[1], "se_10dB_pct_of_genie", 1.0, "Effective SE at 10 dB (% of genie)", (96, 100))):
        for name, member, colour, marker in groups:
            rows = summary[summary["policy"].map(member)]
            axis.scatter(rows["latency_ms_mean"], scale * rows[column], color=colour, marker=marker, s=24,
                         zorder=3, label=name, edgecolors="white", linewidths=0.4)
        axis.set_xscale("log")
        axis.set_xticks([0.5, 1, 2, 5])
        axis.set_xticklabels(["0.5", "1", "2", "5"])
        axis.set_xlim(0.4, 6.5)
        axis.set_ylim(*ylim)
        axis.set_xlabel("Mean beam-training latency per decision (ms)")
        axis.set_ylabel(label)
    reference = summary.set_index("policy")
    for axis, column, scale, tracker_dy in ((axes[0], "exact_top1", 100.0, 0.0),
                                            (axes[1], "se_10dB_pct_of_genie", 1.0, -0.3)):
        for policy, text, dx, dy in (
                ("tracker: window +/-1 + 4 tie partners, full every 8", "tracker, full /8", 1.08, tracker_dy),
                ("full sweep every step", "full sweep", 0.62, 0),
                ("camera + GPS top-4 + tracker, loss-triggered only", "camera + GPS top-4", 1.1, 0)):
            x, y = reference.loc[policy, "latency_ms_mean"], scale * reference.loc[policy, column]
            axis.annotate(text, (x, y), xytext=(x * dx, y + dy), fontsize=6.5, color=INK_MUTED,
                          va="center", ha="left" if dx > 1 else "right")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.07))
    figure.tight_layout()
    save(figure, "fig_side_tracking.png")


def main() -> None:
    figure_system()
    figure_beam_training()
    figure_conformal()
    figure_closed_loop()
    figure_side_information()
    figure_significance()
    figure_side_tracking()
    figure_tracking()
    figure_impairments()
    figure_ris_sizing()


if __name__ == "__main__":
    main()
