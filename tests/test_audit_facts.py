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
