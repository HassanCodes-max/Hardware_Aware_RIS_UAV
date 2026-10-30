import numpy as np

from blind_stacking import BEAMS, fit_weights, shift, shifted_template, subbeam_peak


def test_subbeam_peak_leans_towards_the_stronger_neighbour():
    sweeps = np.full((3, BEAMS), -20.0)
    sweeps[0, 1:4] = (-1.0, 0.0, -1.5)    # stronger left neighbour
    sweeps[1, 0] = 0.0                     # peak on the edge: no interpolation
    sweeps[2, 9:12] = (-1.0, 0.0, -1.0)   # symmetric lobe
    peaks = subbeam_peak(sweeps)
    assert 1.5 < peaks[0] < 2.0
    assert peaks[1] == 0.0
    assert abs(peaks[2] - 10.0) < 1e-9


def test_shifted_template_without_motion_returns_the_last_sweep():
    rng = np.random.default_rng(0)
    sweeps = rng.normal(-10, 3, size=(6, BEAMS))
    sequences = np.zeros(6, dtype=int)
    time_s = np.arange(6) * 0.14
    peaks = subbeam_peak(sweeps)
    out = shifted_template(sweeps, peaks, sequences, time_s, np.array([4]), 1, fit=4, align=1, gain=0.0)
    assert np.allclose(out[0], sweeps[4])


def test_shift_moves_the_lobe_and_stays_causal_per_flight():
    row = np.full(BEAMS, -20.0)
    row[20] = 0.0
    assert shift(row, 3.0).argmax() == 23
    # A new flight starts at sample 3: its template must not use flight 0's sweeps.
    sweeps = np.full((5, BEAMS), -20.0)
    sweeps[:3, 5] = 0.0
    sweeps[3:, 40] = 0.0
    sequences = np.array([0, 0, 0, 1, 1])
    out = shifted_template(sweeps, subbeam_peak(sweeps), sequences, np.arange(5) * 0.14,
                           np.array([3]), 1, fit=8, align=4, gain=0.5)
    assert out[0].argmax() == 40


def test_fit_weights_picks_the_informative_component():
    rng = np.random.default_rng(1)
    n = 400
    target = rng.integers(0, BEAMS, n)
    target_power = np.full((n, BEAMS), 1e-3)
    target_power[np.arange(n), target] = 1.0
    good = rng.normal(size=(n, BEAMS))
    good[np.arange(n), target] += 3.0
    noise = rng.normal(size=(n, BEAMS))
    weights = fit_weights({"noise": noise, "good": good}, "noise", np.arange(n), target, target_power)
    assert weights["good"] > weights["noise"]
