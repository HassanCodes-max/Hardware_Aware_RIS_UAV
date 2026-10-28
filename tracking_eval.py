"""Predictive RIS phase tracking: position predictors scored by array gain.

The first paper draft scored trackers by mean absolute per-element phase
error and concluded that open-loop, position-based RIS control is
impossible at sub-THz. That metric counts a common phase offset on all
elements as error, but a common offset does not change received power at
all. What matters is the fraction of the ideal coherent array gain that the
predicted phase profile delivers at the true position:

    G = |sum_n g_n e^{j phi_hat_n}|^2 / (sum_n |g_n|)^2

This script reports both metrics for the same predictors so the difference
is visible. All predictors see the same noisy position measurements; none
sees ground truth.

Run:  python tracking_eval.py
Writes results/tracking_metrics.json and results/tracking_metrics.csv
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from channel import beamforming_gain_ratio, optimal_ris_phases, ris_element_channels
from config import SimulationConfig
from dataset_generator import generate_designed_trajectories, generate_random_trajectories

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MEASUREMENT_NOISE_M = 0.02
WINDOW = 8
EVAL_STRIDE = 10
# Prediction horizons in steps of 10 ms: the RIS must be configured for
# where the UAV will be once measurement, compute and control latency pass.
HORIZONS = (1, 10, 50)


def noisy_measurements(
    positions_m: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add i.i.d. Gaussian position noise (RTK-class sensor)."""

    return positions_m + rng.normal(0.0, MEASUREMENT_NOISE_M, positions_m.shape)


def kalman_predict(
    measurements_m: np.ndarray,
    dt_s: float,
    accel_psd: float,
    horizon: int = 1,
) -> np.ndarray:
    """
    Constant-velocity Kalman filter; returns the horizon-step-ahead
    prediction made after each measurement (row t predicts t + horizon).
    """

    f = np.array([[1.0, dt_s], [0.0, 1.0]])
    q = accel_psd * np.array([[dt_s**3 / 3.0, dt_s**2 / 2.0], [dt_s**2 / 2.0, dt_s]])
    f_horizon = np.linalg.matrix_power(f, horizon)
    h = np.array([[1.0, 0.0]])
    r = MEASUREMENT_NOISE_M**2

    predictions = np.zeros_like(measurements_m)

    # The three axes are independent with this model.
    for axis in range(3):
        state = np.array([measurements_m[0, axis], 0.0])
        covariance = np.diag([r, 100.0])

        for t, z in enumerate(measurements_m[:, axis]):
            innovation_cov = h @ covariance @ h.T + r
            gain = covariance @ h.T / innovation_cov
            state = state + (gain * (z - h @ state)).ravel()
            covariance = (np.eye(2) - gain @ h) @ covariance

            predictions[t, axis] = (f_horizon @ state)[0]

            state = f @ state
            covariance = f @ covariance @ f.T + q

    return predictions


def window_features(
    measurements_m: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Stack the last WINDOW measurements relative to the newest one."""

    rows = []
    for t in range(WINDOW - 1, len(measurements_m) - horizon):
        window = measurements_m[t - WINDOW + 1:t + 1]
        rows.append((window - window[-1]).ravel())

    return np.asarray(rows)


def window_targets(
    measurements_m: np.ndarray,
    positions_m: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return the true future position relative to the newest measurement."""

    return positions_m[WINDOW - 1 + horizon:] - measurements_m[WINDOW - 1:-horizon]


def predict_all(
    measurements_m: np.ndarray,
    dt_s: float,
    accel_psd: float,
    learned: Ridge,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Return predictions aligned with positions_m[WINDOW - 1 + horizon:]."""

    current = measurements_m[WINDOW - 1:-horizon]
    previous = measurements_m[WINDOW - 2:-horizon - 1]
    kalman = kalman_predict(measurements_m, dt_s, accel_psd, horizon)

    return {
        "persistence": current,
        "constant_velocity": current + horizon * (current - previous),
        "kalman_cv": kalman[WINDOW - 1:-horizon],
        "learned_ridge": current + learned.predict(window_features(measurements_m, horizon)),
    }


def score(
    predicted_m: np.ndarray,
    truth_m: np.ndarray,
    config: SimulationConfig,
) -> dict[str, float]:
    """Score predictions by position error, phase error and array gain."""

    gains = []
    phase_errors = []

    for estimate, actual in zip(predicted_m[::EVAL_STRIDE], truth_m[::EVAL_STRIDE]):
        true_bs, true_uav = ris_element_channels(actual, config)
        if not np.any(np.abs(true_uav)):
            continue
        est_bs, est_uav = ris_element_channels(estimate, config)

        applied = optimal_ris_phases(est_bs, est_uav)
        ideal = optimal_ris_phases(true_bs, true_uav)

        gains.append(beamforming_gain_ratio(applied, true_bs, true_uav))
        phase_errors.append(np.mean(np.abs(np.angle(np.exp(1j * (applied - ideal))))))

    gains = np.asarray(gains)
    errors = np.linalg.norm(predicted_m - truth_m, axis=1)
    evaluated = len(truth_m[::EVAL_STRIDE])

    if len(gains) == 0:
        return {
            "position_rmse_cm": float(100.0 * np.sqrt(np.mean(errors**2))),
            "served_fraction_pct": 0.0,
        }

    return {
        "position_rmse_cm": float(100.0 * np.sqrt(np.mean(errors**2))),
        "served_fraction_pct": float(100.0 * len(gains) / evaluated),
        "mean_abs_phase_error_deg": float(np.degrees(np.mean(phase_errors))),
        "mean_array_gain_pct": float(100.0 * gains.mean()),
        "p5_array_gain_pct": float(100.0 * np.percentile(gains, 5)),
        "mean_gain_loss_db": float(-10.0 * np.log10(gains.mean())),
    }


def tune_kalman(
    training: list,
    dt_s: float,
    rng: np.random.Generator,
) -> float:
    """Pick the acceleration PSD with the lowest one-step RMSE on training data."""

    candidates = [0.1, 1.0, 10.0, 100.0, 1000.0]
    errors = []

    for psd in candidates:
        squared = []
        for _, _, trajectory in training[:20]:
            z = noisy_measurements(trajectory.positions_m, rng)
            predicted = kalman_predict(z, dt_s, psd)[:-1]
            squared.append(np.sum((predicted - trajectory.positions_m[1:]) ** 2, axis=1))
        errors.append(np.mean(np.concatenate(squared)))

    return candidates[int(np.argmin(errors))]


def main() -> None:
    """Train on random trajectories, evaluate on the designed scenarios."""

    config = SimulationConfig().ideal()
    dt_s = config.time_step_s
    rng = np.random.default_rng(0)

    training = generate_random_trajectories(config, 60, seed=123, starting_id=100)
    testing = generate_designed_trajectories(config)

    accel_psd = tune_kalman(training, dt_s, rng)

    training_measurements = [
        (noisy_measurements(trajectory.positions_m, rng), trajectory.positions_m)
        for _, _, trajectory in training
    ]

    rows = []
    for horizon in HORIZONS:
        learned = Ridge(alpha=1e-3).fit(
            np.vstack([window_features(z, horizon) for z, _ in training_measurements]),
            np.vstack([window_targets(z, p, horizon) for z, p in training_measurements]),
        )

        for _, trajectory_type, trajectory in testing:
            z = noisy_measurements(trajectory.positions_m, rng)
            truth = trajectory.positions_m[WINDOW - 1 + horizon:]

            started = time.perf_counter()
            predictions = predict_all(z, dt_s, accel_psd, learned, horizon)
            elapsed_ms_per_step = (time.perf_counter() - started) * 1e3 / len(truth)

            for method, predicted in predictions.items():
                rows.append({
                    "horizon_ms": horizon * dt_s * 1e3,
                    "trajectory_type": trajectory_type,
                    "method": method,
                    **score(predicted, truth, config),
                })

        print(f"horizon {horizon * dt_s * 1e3:.0f} ms done "
              f"({elapsed_ms_per_step:.3f} ms/step for all predictors)")

    table = pd.DataFrame(rows)
    summary = (
        table.drop(columns="trajectory_type")
        .groupby(["horizon_ms", "method"])
        .mean()
        .round(3)
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "tracking_metrics.csv", index=False)
    (RESULTS_DIR / "tracking_metrics.json").write_text(json.dumps({
        "measurement_noise_cm": 100 * MEASUREMENT_NOISE_M,
        "ris_elements": config.ris_elements,
        "kalman_accel_psd": accel_psd,
        "summary": [
            {"horizon_ms": h, "method": m, **values}
            for (h, m), values in summary.to_dict(orient="index").items()
        ],
    }, indent=2))

    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()
