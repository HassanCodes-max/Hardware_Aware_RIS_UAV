"""Tests for significance_tests.py and the SCENARIO23_FEATURES switch.

Run from the repository root:  python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import beam_prediction  # noqa: E402
from significance_tests import holm, paired_test  # noqa: E402


def test_holm_is_monotone_and_capped():
    adjusted = holm([0.01, 0.04, 0.03, 0.5])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_paired_test_counts_and_sign():
    rng = np.random.default_rng(0)
    a = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    b = a - np.array([0.1, 0.1, 0.0, 0.2, 0.1, 0.05])
    result = paired_test(a, b, np.ones(6), rng)
    assert (result["wins"], result["ties"], result["losses"]) == (5, 1, 0)
    assert result["difference"] == pytest.approx(0.55 / 6)
    assert result["ci95_low"] >= 0.0


def test_identical_methods_are_not_significant():
    a = np.linspace(0.5, 1.0, 10)
    result = paired_test(a, a.copy(), np.ones(10), np.random.default_rng(0))
    assert result["wilcoxon_p"] == 1.0 and result["difference"] == 0.0


def test_unknown_feature_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("SCENARIO23_FEATURES", "lidar")
    with pytest.raises(ValueError):
        beam_prediction.load_scenario23()


def test_power_mode_has_no_side_information(monkeypatch):
    monkeypatch.setenv("SCENARIO23_FEATURES", "power")
    power, labels, sequences, time_s, side = beam_prediction.load_scenario23()
    assert side is None and power.shape == (len(labels), beam_prediction.BEAMS)
