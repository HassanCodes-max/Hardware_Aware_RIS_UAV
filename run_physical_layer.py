"""Physical-layer experiments: link budget, impairments, scaling, estimation.

Run:  python run_physical_layer.py
Writes results/physical_layer_metrics.json
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from channel import estimate_cascaded_channel, link_metrics, ris_element_channels
from config import SimulationConfig
from hardware import hardware_aware_phases
from trajectory import straight_line_trajectory

RESULTS_DIR = Path(__file__).resolve().parent / "results"

TRAJECTORY_STRIDE = 10
# QPSK needs about 9.8 dB SNR for BER 1e-3.
QPSK_1E3_SNR_DB = 9.8

IMPAIRMENT_SWITCHES = {
    "pa_nonlinearity": "use_pa_nonlinearity",
    "ris_phase_quantization": "use_ris_phase_quantization",
    "ris_phase_noise": "use_ris_phase_noise",
    "ris_amplitude_phase_coupling": "use_ris_amplitude_phase_coupling",
    "s11_mismatch": "use_s11_mismatch",
}


def legacy_reproduction() -> dict:
    """Check that the v1 preset reproduces the first draft's SNR figures."""

    config = SimulationConfig.paper_v1()

    return {
        f"snr_db_at_{x:.0f}m": round(link_metrics(np.array([x, 0.0, 20.0]), config)["snr_db"], 2)
        for x in (20.0, 170.0, 320.0)
    }


def trajectory_comparison(
    config: SimulationConfig,
) -> dict:
    """Compare ideal, naive and hardware-aware control along the default flight."""

    trajectory = straight_line_trajectory(config)
    positions = trajectory.positions_m[::TRAJECTORY_STRIDE]

    rows = {"ideal": [], "hardware_naive": [], "hardware_aware": []}

    for position in positions:
        rows["ideal"].append(link_metrics(position, config.ideal()))
        rows["hardware_naive"].append(link_metrics(position, config))

        bs_to_ris, ris_to_uav = ris_element_channels(position, config)
        phases = hardware_aware_phases(bs_to_ris * ris_to_uav, config)
        rows["hardware_aware"].append(link_metrics(position, config, phases))

    summary = {}
    for name, metrics in rows.items():
        sndr = np.array([m["sndr_db"] for m in metrics])
        summary[name] = {
            "mean_sndr_db": round(float(sndr.mean()), 2),
            "sndr_db_at_start": round(float(sndr[0]), 2),
            "sndr_db_at_end": round(float(sndr[-1]), 2),
            "mean_spectral_efficiency_bps_hz": round(
                float(np.mean([m["spectral_efficiency_bps_hz"] for m in metrics])), 3
            ),
            "availability_sndr_ge_0db_pct": round(float(100.0 * np.mean(sndr >= 0.0)), 1),
            "availability_qpsk_ber_1e-3_pct": round(
                float(100.0 * np.mean(sndr >= QPSK_1E3_SNR_DB)), 1
            ),
            "median_ber_qpsk": float(np.median([m["ber_qpsk"] for m in metrics])),
        }

    summary["trajectory"] = {
        "start_m": positions[0].tolist(),
        "end_m": positions[-1].tolist(),
        "samples": len(positions),
    }

    return summary


def impairment_ablation(
    config: SimulationConfig,
    position: np.ndarray,
) -> dict:
    """SNDR penalty of each impairment alone, relative to ideal hardware."""

    ideal_db = link_metrics(position, config.ideal())["sndr_db"]
    penalties = {}

    for name, switch in IMPAIRMENT_SWITCHES.items():
        single = replace(config.ideal(), **{switch: True})
        penalties[name] = round(ideal_db - link_metrics(position, single)["sndr_db"], 2)

    penalties["all_combined"] = round(ideal_db - link_metrics(position, config)["sndr_db"], 2)

    return {"position_m": position.tolist(), "ideal_snr_db": round(ideal_db, 2),
            "penalty_db": penalties}


def element_scaling(
    config: SimulationConfig,
    position: np.ndarray,
) -> dict:
    """Received power versus RIS size, ideal hardware, at 60 and 140 GHz."""

    results = {}
    for frequency in (60e9, 140e9):
        powers = {}
        for side in (8, 16, 32, 64):
            sized = replace(config.ideal(), ris_rows=side, ris_columns=side,
                            carrier_frequency_hz=frequency)
            power_w = link_metrics(position, sized)["received_power_w"]
            powers[f"{side}x{side}"] = round(10.0 * np.log10(power_w * 1e3), 2)
        results[f"{frequency / 1e9:.0f}GHz_received_dbm"] = powers

    return results


def ris_sizing(
    config: SimulationConfig,
    sides: tuple = (32, 64, 96, 128),
) -> dict:
    """
    Link availability (SNDR >= 0 dB) along the default flight versus RIS size.

    Answers the design question "how large must the RIS be?" instead of
    tuning geometry until a percentage looks good.
    """

    positions = straight_line_trajectory(config).positions_m[::TRAJECTORY_STRIDE * 5]
    results = {}

    for side in sides:
        sized = replace(config, ris_rows=side, ris_columns=side)
        sndr_ideal, sndr_aware = [], []
        for position in positions:
            sndr_ideal.append(link_metrics(position, sized.ideal())["sndr_db"])
            bs_to_ris, ris_to_uav = ris_element_channels(position, sized)
            phases = hardware_aware_phases(bs_to_ris * ris_to_uav, sized)
            sndr_aware.append(link_metrics(position, sized, phases)["sndr_db"])

        results[f"{side}x{side}"] = {
            "aperture_cm": round(100.0 * side * sized.ris_element_spacing_m, 1),
            "ideal_availability_pct": round(float(100.0 * np.mean(np.array(sndr_ideal) >= 0.0)), 1),
            "hardware_aware_availability_pct": round(
                float(100.0 * np.mean(np.array(sndr_aware) >= 0.0)), 1
            ),
        }

    return results


def estimation_study(
    config: SimulationConfig,
    distances_m: tuple = (20.0, 50.0, 100.0),
    pilot_lengths: tuple = (256, 1024, 4096),
    trials: int = 3,
) -> list[dict]:
    """NMSE of naive LS vs hardware-aware LMMSE cascaded-channel estimation."""

    rows = []
    for distance in distances_m:
        position = np.array([distance, 0.0, 30.0])
        for pilots in pilot_lengths:
            row = {"x_m": distance, "pilot_symbols_per_pattern": pilots}
            for label, aware in (("naive_ls_nmse_db", False), ("hardware_aware_lmmse_nmse_db", True)):
                row[label] = round(float(np.mean([
                    estimate_cascaded_channel(
                        position, config, pilots, aware, np.random.default_rng(trial)
                    )["nmse_db"]
                    for trial in range(trials)
                ])), 2)
            rows.append(row)

    return rows


def main() -> None:
    """Run every experiment and write one JSON file."""

    config = SimulationConfig()
    reference = np.array([50.0, 0.0, 30.0])

    results = {
        "config": {
            "carrier_ghz": config.carrier_frequency_hz / 1e9,
            "ris": f"{config.ris_rows}x{config.ris_columns}",
            "bs_gain_dbi": config.bs_antenna_gain_dbi,
            "uav_gain_dbi": config.uav_antenna_gain_dbi,
            "tx_power_dbm": 10.0 * np.log10(config.transmit_power_w * 1e3),
            "noise_dbm": round(10.0 * np.log10(config.noise_power_w * 1e3), 2),
        },
        "legacy_v1_reproduction": legacy_reproduction(),
        "trajectory_comparison": trajectory_comparison(config),
        "impairment_ablation": impairment_ablation(config, reference),
        "element_scaling": element_scaling(config, reference),
        "ris_sizing_availability": ris_sizing(config),
        "channel_estimation": estimation_study(config),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "physical_layer_metrics.json"
    output.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
