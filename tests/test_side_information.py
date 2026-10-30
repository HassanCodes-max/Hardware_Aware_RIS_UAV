"""Tests for the UAV locator and side-information candidates in the closed-loop tracker.

Run from the repository root:  python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beam_prediction import BEAMS  # noqa: E402
from camera_beam_prediction import locate_uav  # noqa: E402
from closed_loop_beam_tracking import Policy, run_flight  # noqa: E402


def test_locate_uav_finds_dark_dot_on_sky_gradient_with_glare(tmp_path):
    height, width = 540, 960
    yy, xx = np.mgrid[0:height, 0:width]
    sky = 120 + 60 * xx / width + 20 * yy / height          # smooth gradient
    # Sun glare as in the real frames: a bright, soft-edged streak. (A hard-edged
    # bright disc would leave a dark halo after the sky is subtracted; the real
    # frames do not show one, see camera_beam_prediction.locate_uav.)
    sky += 120 * np.exp(-((xx - 0.6 * yy - 700) / 25.0) ** 2)
    sky[300:303, 412:415] -= 40                            # UAV: 3x3 dark blob
    path = tmp_path / "frame.png"
    Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8)).convert("RGB").save(path)

    x, y, contrast = locate_uav(path)
    assert abs(x * width - 413) <= 1 and abs(y * height - 301) <= 1
    assert contrast > 20


def test_side_ranking_adds_candidates_and_catches_a_jump():
    steps = 3
    power = np.full((steps, BEAMS), 1.0)
    power[0, 10] = 10.0
    power[1, 40] = 10.0          # far jump: not in the +/-1 window around beam 10
    power[1, 10] = 9.0           # small drop: no beam-loss fallback
    power[2, 40] = 10.0
    graph = np.tile(np.arange(BEAMS), (BEAMS, 1))
    side = np.tile(np.arange(BEAMS), (steps, 1))
    side[1, :2] = [40, 41]

    without = run_flight(power, Policy("p", None, window=1), graph)
    with_side = run_flight(power, Policy("p", None, window=1, side=2), graph, side)
    assert without["chosen"][1] == 10
    assert with_side["chosen"][1] == 40
    assert with_side["beams"][1] == without["beams"][1] + 2
