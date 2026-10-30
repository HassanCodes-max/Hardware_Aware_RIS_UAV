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
from parametric_estimation import (
    cascaded_channel,
    estimate_from_prior,
    random_training,
    refine_position,
    send_pilots,
)
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


def closed_loop_study(
    config: SimulationConfig,
    distances_m: tuple = (20.0, 50.0, 100.0),
    pilot_lengths: tuple = (256, 1024, 4096),
    trials: int = 3,
) -> list[dict]:
    """
    Estimate the channel, configure the RIS from the estimate, measure SNDR.

    Pipelines:
      - naive: LS estimate, phases = -angle(estimate) (hardware-unaware);
      - hardware_aware: LMMSE estimate, coupling/quantization-aware phases.
    Both are compared with perfect-CSI hardware-aware control, so the gap
    is the cost of estimation alone.
    """

    rows = []
    for distance in distances_m:
        position = np.array([distance, 0.0, 30.0])
        bs_to_ris, ris_to_uav = ris_element_channels(position, config)
        perfect_db = link_metrics(
            position, config, hardware_aware_phases(bs_to_ris * ris_to_uav, config)
        )["sndr_db"]

        for pilots in pilot_lengths:
            sndr = {"naive": [], "hardware_aware": []}
            for trial in range(trials):
                rng = np.random.default_rng(trial)
                naive = estimate_cascaded_channel(position, config, pilots, False, rng)["estimate"]
                aware = estimate_cascaded_channel(position, config, pilots, True, rng)["estimate"]
                sndr["naive"].append(link_metrics(position, config, -np.angle(naive))["sndr_db"])
                sndr["hardware_aware"].append(
                    link_metrics(position, config, hardware_aware_phases(aware, config))["sndr_db"]
                )

            rows.append({
                "x_m": distance,
                "pilot_symbols_per_pattern": pilots,
                "perfect_csi_sndr_db": round(perfect_db, 2),
                "naive_sndr_db": round(float(np.mean(sndr["naive"])), 2),
                "hardware_aware_sndr_db": round(float(np.mean(sndr["hardware_aware"])), 2),
            })

    return rows


def parametric_study(
    config: SimulationConfig,
    distances_m: tuple = (20.0, 50.0, 100.0),
    gps_error_std_m: float = 5.0,
    budgets: tuple = ((16, 256), (64, 256), (64, 1024)),
    trials: int = 5,
) -> list[dict]:
    """
    Position-parametric estimation vs per-element LMMSE, by pilot budget.

    Parametric: coarse GPS prior (gps_error_std_m per axis), M random
    patterns of L pilot symbols, direction refinement, hardware-aware
    phases from g(p_hat). Compared with GPS-only control and with
    per-element LMMSE using all N patterns.
    """

    rows = []
    for distance in distances_m:
        position = np.array([distance, 0.0, 30.0])
        perfect_db = link_metrics(
            position, config, hardware_aware_phases(cascaded_channel(position, config), config)
        )["sndr_db"]

        def configure_from(estimate_position):
            return link_metrics(
                position, config,
                hardware_aware_phases(cascaded_channel(estimate_position, config), config),
            )["sndr_db"]

        rngs = [np.random.default_rng(100 + trial) for trial in range(trials)]
        priors = [position + rng.normal(0.0, gps_error_std_m, 3) for rng in rngs]

        row = {
            "x_m": distance,
            "perfect_csi_sndr_db": round(perfect_db, 2),
            "gps_only_sndr_db": round(float(np.mean([configure_from(p) for p in priors])), 2),
        }

        for patterns, symbols in budgets:
            sndr = []
            for rng, prior in zip(rngs, priors):
                received, nominal = send_pilots(
                    position, random_training(patterns, config, rng), config, symbols, rng
                )
                sndr.append(configure_from(refine_position(prior, received, nominal, config)))
            row[f"parametric_M{patterns}_L{symbols}_sndr_db"] = round(float(np.mean(sndr)), 2)
            row[f"parametric_M{patterns}_L{symbols}_pilot_symbols"] = patterns * symbols

        row["lmmse_N_L4096_pilot_symbols"] = config.ris_elements * 4096
        rows.append(row)

    return rows


def adaptive_study(
    config: SimulationConfig,
    distances_m: tuple = (20.0, 50.0, 100.0, 200.0),
    priors: tuple = (("gps_5m", 5.0), ("kalman_500ms", 0.31), ("kalman_100ms", 0.05)),
    trials: int = 10,
) -> list[dict]:
    """
    Adaptive parametric estimation: the pilot budget and search width are
    set from the prior and the predicted SNR. Prior std values come from
    GPS and from the Kalman RMSE in tracking_eval.py at 500 / 100 ms.
    Reports the delivered fraction of perfect-CSI SNR.
    """

    rows = []
    for distance in distances_m:
        position = np.array([distance, 0.0, 30.0])
        perfect = link_metrics(
            position, config, hardware_aware_phases(cascaded_channel(position, config), config)
        )["sndr_linear"]

        for name, std in priors:
            ratios, pilots = [], []
            for trial in range(trials):
                rng = np.random.default_rng(500 + trial)
                prior = position + rng.normal(0.0, std, 3)
                estimate, used = estimate_from_prior(position, prior, std, config, rng)
                achieved = link_metrics(
                    position, config, hardware_aware_phases(cascaded_channel(estimate, config), config)
                )["sndr_linear"]
                ratios.append(achieved / perfect)
                pilots.append(used)

            rows.append({
                "x_m": distance,
                "prior": name,
                "mean_pct_of_perfect_csi_snr": round(100.0 * float(np.mean(ratios)), 2),
                "worst_pct_of_perfect_csi_snr": round(100.0 * float(np.min(ratios)), 2),
                "mean_pilot_symbols": int(np.mean(pilots)),
                "pilot_saving_vs_lmmse_x": round(
                    config.ris_elements * 4096 / max(float(np.mean(pilots)), 1.0), 1
                ),
            })

    return rows


def phase_resolution_study(
    config: SimulationConfig,
    position: np.ndarray,
    bits: tuple = (1, 2, 3, 4),
) -> list[dict]:
    """
    SNDR vs RIS phase resolution: ideal, naive control, hardware-aware control.

    Shows how much coupling-aware selection recovers as more realisable
    states become available.
    """

    ideal_db = link_metrics(position, config.ideal())["sndr_db"]
    rows = []

    for b in bits:
        quantized = replace(config, ris_phase_bits=b)
        bs_to_ris, ris_to_uav = ris_element_channels(position, quantized)
        aware = hardware_aware_phases(bs_to_ris * ris_to_uav, quantized)
        naive_db = link_metrics(position, quantized)["sndr_db"]
        aware_db = link_metrics(position, quantized, aware)["sndr_db"]
        rows.append({
            "bits": b,
            "ideal_sndr_db": round(ideal_db, 2),
            "naive_sndr_db": round(naive_db, 2),
            "hardware_aware_sndr_db": round(aware_db, 2),
            "aware_gain_db": round(aware_db - naive_db, 2),
        })

    return rows


def carrier_study(
    config: SimulationConfig,
    aperture_m: float = 0.08,
) -> list[dict]:
    """
    60 vs 140 GHz at equal RIS aperture, full impairments and ITU absorption.

    Equal aperture means more elements at the higher carrier (lambda/2
    spacing), which offsets the larger per-element path loss.
    """

    from atmosphere import specific_gaseous_attenuation_db_per_km

    positions = straight_line_trajectory(config).positions_m[::TRAJECTORY_STRIDE * 5]
    rows = []

    # D-band hardware is weaker: about 20 dBm PA saturation (InP, IMS 2020)
    # -> 14 dBm average at 6 dB back-off, and a conservative 10 dB receiver
    # noise figure (measured D-band SiGe LNAs alone: 4.8-6.1 dB; the receiver
    # adds front-end loss and later stages). See the NF sensitivity below.
    cases = (
        ("60 GHz", 60e9, {}),
        ("140 GHz, 60 GHz-class hardware", 140e9, {}),
        ("140 GHz, D-band hardware", 140e9, {"transmit_power_w": 10 ** (14 / 10) / 1e3, "noise_figure_db": 10.0}),
    )

    for label, frequency, hardware in cases:
        side = int(round(aperture_m / (299_792_458.0 / frequency / 2.0)))
        carrier = replace(config, carrier_frequency_hz=frequency, ris_rows=side, ris_columns=side, **hardware)

        ideal, aware = [], []
        for position in positions:
            ideal.append(link_metrics(position, carrier.ideal())["sndr_db"])
            bs_to_ris, ris_to_uav = ris_element_channels(position, carrier)
            aware.append(link_metrics(
                position, carrier, hardware_aware_phases(bs_to_ris * ris_to_uav, carrier)
            )["sndr_db"])

        rows.append({
            "case": label,
            "ris": f"{side}x{side}",
            "gaseous_absorption_db_per_km": round(specific_gaseous_attenuation_db_per_km(
                frequency, config.atmospheric_pressure_hpa, config.atmospheric_temperature_k,
                config.water_vapour_density_g_m3), 3),
            "ideal_sndr_db_at_start": round(ideal[0], 2),
            "hardware_aware_sndr_db_at_start": round(aware[0], 2),
            "ideal_availability_pct": round(float(100.0 * np.mean(np.array(ideal) >= 0.0)), 1),
            "hardware_aware_availability_pct": round(float(100.0 * np.mean(np.array(aware) >= 0.0)), 1),
        })

    return rows


def coupling_sensitivity(
    config: SimulationConfig,
    position: np.ndarray,
    beta_mins: tuple = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> list[dict]:
    """
    Impairment penalty and aware-control gain vs coupling strength.

    beta_min = 1 means no coupling. No measured 60 GHz value exists, so
    the conclusions must hold across this range rather than at one point.
    """

    ideal_db = link_metrics(position, config.ideal())["sndr_db"]
    rows = []

    for beta_min in beta_mins:
        coupled = replace(config, ris_coupling_beta_min=beta_min)
        bs_to_ris, ris_to_uav = ris_element_channels(position, coupled)
        aware_db = link_metrics(position, coupled, hardware_aware_phases(bs_to_ris * ris_to_uav, coupled))["sndr_db"]
        naive_db = link_metrics(position, coupled)["sndr_db"]
        rows.append({
            "beta_min": beta_min,
            "total_penalty_naive_db": round(ideal_db - naive_db, 2),
            "total_penalty_aware_db": round(ideal_db - aware_db, 2),
            "aware_gain_db": round(aware_db - naive_db, 2),
        })

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
        "closed_loop_estimate_then_control": closed_loop_study(config),
        "parametric_estimation": parametric_study(config),
        "adaptive_parametric_estimation": adaptive_study(config),
        "phase_resolution": phase_resolution_study(config, reference),
        "coupling_sensitivity": coupling_sensitivity(config, reference),
        "carrier_comparison_equal_aperture": carrier_study(config),
        "ris_sizing_140ghz_dband_hardware": ris_sizing(
            replace(config, carrier_frequency_hz=140e9,
                    transmit_power_w=10 ** (14 / 10) / 1e3, noise_figure_db=10.0),
            sides=(75, 150, 225, 300),
        ),
        # The 140 GHz RIS size hinges on the receiver noise figure: 6 dB is a
        # measured D-band LNA plus ~1 dB, 7 dB the 60 GHz value, 10 dB the default.
        "ris_sizing_140ghz_nf_sensitivity": {
            f"nf_{nf:g}db": ris_sizing(
                replace(config, carrier_frequency_hz=140e9,
                        transmit_power_w=10 ** (14 / 10) / 1e3, noise_figure_db=nf),
                sides=(150, 188, 225, 263, 300),
            )
            for nf in (6.0, 7.0, 10.0)
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "physical_layer_metrics.json"
    output.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
