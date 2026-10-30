"""Regression tests for the physics model and the facts recorded in AUDIT.md.

Run from the repository root:  python -m pytest tests
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from channel import (  # noqa: E402
    beamforming_gain_ratio,
    link_metrics,
    optimal_ris_phases,
    ris_element_channels,
)
from config import SimulationConfig  # noqa: E402
from ris import ris_element_positions  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "public_data" / "scenario23_dev"


def test_paper_v1_preset_reproduces_first_draft_snr():
    snr = link_metrics(np.array([20.0, 0.0, 20.0]), SimulationConfig.paper_v1())["snr_db"]
    assert abs(snr - (-43.23)) < 0.05


def test_coherent_gain_scales_with_n_squared():
    config = SimulationConfig().ideal()
    position = np.array([50.0, 0.0, 30.0])
    small = link_metrics(position, replace(config, ris_rows=16, ris_columns=16))
    large = link_metrics(position, replace(config, ris_rows=32, ris_columns=32))
    assert abs((large["snr_db"] - small["snr_db"]) - 12.04) < 0.1


def test_flat_panel_layout_matches_original():
    config = SimulationConfig.paper_v1()
    positions = ris_element_positions(config)
    assert np.allclose(positions[:, 2], 20.0)
    assert positions[1, 0] > positions[0, 0]


def test_common_phase_offset_costs_no_gain():
    config = SimulationConfig().ideal()
    bs_to_ris, ris_to_uav = ris_element_channels(np.array([40.0, 5.0, 30.0]), config)
    phases = optimal_ris_phases(bs_to_ris, ris_to_uav)
    assert beamforming_gain_ratio(phases + 1.7, bs_to_ris, ris_to_uav) > 0.999


def test_uav_behind_panel_gets_no_signal():
    metrics = link_metrics(np.array([-20.0, 0.0, 30.0]), SimulationConfig())
    assert metrics["received_power_w"] == 0.0


def test_impairments_never_help():
    position = np.array([50.0, 0.0, 30.0])
    config = SimulationConfig()
    assert link_metrics(position, config)["sndr_db"] < link_metrics(position, config.ideal())["snr_db"]


def test_scenario23_label_is_argmax_of_power():
    beams = pd.read_csv(DATA / "scenario23_beam_dataset.csv")
    power = beams[[f"power_{i}" for i in range(1, 65)]].to_numpy()
    assert (power.argmax(axis=1) + 1 == beams["beam_label"].to_numpy()).all()


def test_kinematics_loader_reads_unit2_files(tmp_path):
    from public_data_adapter import load_scenario23_kinematics

    (tmp_path / "unit1").mkdir()
    (tmp_path / "unit2").mkdir()
    (tmp_path / "unit1" / "bs.txt").write_text("33.0000\n-111.0000\n")
    rows = []
    for i, (lat, lon) in enumerate([(33.0001, -111.0), (33.0, -110.9999)]):
        (tmp_path / "unit2" / f"loc{i}.txt").write_text(f"{lat} {lon}")
        (tmp_path / "unit2" / f"h{i}.txt").write_text("[20.0]")
        rows.append({"unit1_loc": "./unit1/bs.txt", "unit2_loc": f"./unit2/loc{i}.txt",
                     "unit2_height": f"./unit2/h{i}.txt"})

    kinematics = load_scenario23_kinematics(tmp_path, pd.DataFrame(rows))

    assert kinematics["uav_north_m"].iloc[0] > 10.0
    assert kinematics["uav_east_m"].iloc[1] > 8.0
    assert np.allclose(kinematics["unit2_height"], 20.0 * 0.3048)   # stored in feet
    assert (kinematics["uav_elevation_rad"] > 0).all()


def test_parametric_estimator_recovers_perfect_csi_gain():
    from hardware import hardware_aware_phases
    from parametric_estimation import cascaded_channel, random_training, refine_position, send_pilots

    config = SimulationConfig()
    position = np.array([50.0, 0.0, 30.0])
    rng = np.random.default_rng(0)
    received, nominal = send_pilots(position, random_training(64, config, rng), config, 1024, rng)
    estimate = refine_position(position + rng.normal(0.0, 5.0, 3), received, nominal, config)

    def sndr(p):
        return link_metrics(position, config, hardware_aware_phases(cascaded_channel(p, config), config))["sndr_db"]

    assert sndr(estimate) > sndr(position) - 0.5


def test_measure_and_pick_keeps_strongest_measured_beam():
    from beam_training_overhead import adaptive_sizes, measure_and_pick

    power = np.array([[1.0, 5.0, 3.0, 9.0], [4.0, 1.0, 2.0, 3.0]])
    ranking = np.array([[2, 1, 0, 3], [3, 2, 0, 1]])

    picked = measure_and_pick(ranking, np.array([2, 3]), power)

    # Row 0 measures beams 2 and 1 only, so the global best (3) is out of reach.
    assert picked.tolist() == [1, 0]
    assert adaptive_sizes(np.array([[0.6, 0.3, 0.1, 0.0]]), 0.85).tolist() == [2]
