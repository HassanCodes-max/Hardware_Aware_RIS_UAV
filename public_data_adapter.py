
"""Utilities for importing public UAV trajectory data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PublicTrajectory:
    """Trajectory extracted from a public dataset."""

    trajectory_id: str
    time_s: np.ndarray
    positions_m: np.ndarray
    velocities_m_s: np.ndarray


def load_csv(
    csv_path: Path,
) -> pd.DataFrame:
    """Load a public dataset CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}"
        )

    return pd.read_csv(csv_path)


def inspect_columns(
    csv_path: Path,
) -> None:
    """Print the columns available in a public CSV."""

    dataframe = load_csv(csv_path)

    print(
        "\nPublic dataset columns:\n"
    )

    for column in dataframe.columns:
        print(column)

    print(
        f"\nRows: {len(dataframe)}"
    )


def latitude_longitude_to_local_xy(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude into a local approximate XY system.

    The first coordinate becomes the local origin.
    Distances are approximated using a local tangent-plane model.
    """

    latitude_deg = np.asarray(
        latitude_deg,
        dtype=float,
    )

    longitude_deg = np.asarray(
        longitude_deg,
        dtype=float,
    )

    earth_radius_m = 6_371_000.0

    latitude_rad = np.radians(
        latitude_deg
    )

    longitude_rad = np.radians(
        longitude_deg
    )

    latitude_reference_rad = (
        latitude_rad[0]
    )

    longitude_reference_rad = (
        longitude_rad[0]
    )

    x_m = (
        earth_radius_m
        * (
            longitude_rad
            - longitude_reference_rad
        )
        * np.cos(
            latitude_reference_rad
        )
    )

    y_m = (
        earth_radius_m
        * (
            latitude_rad
            - latitude_reference_rad
        )
    )

    return x_m, y_m


def estimate_velocity(
    positions_m: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Estimate velocity from consecutive positions."""

    positions_m = np.asarray(
        positions_m,
        dtype=float,
    )

    time_s = np.asarray(
        time_s,
        dtype=float,
    )

    velocities_m_s = np.zeros_like(
        positions_m
    )

    if len(positions_m) < 2:
        return velocities_m_s

    for index in range(
        1,
        len(positions_m),
    ):

        delta_time = (
            time_s[index]
            - time_s[index - 1]
        )

        if delta_time <= 0:
            continue

        velocities_m_s[index] = (
            (
                positions_m[index]
                - positions_m[index - 1]
            )
            / delta_time
        )

    velocities_m_s[0] = (
        velocities_m_s[1]
    )

    return velocities_m_s


def build_public_trajectory(
    trajectory_id: str,
    time_s: np.ndarray,
    positions_m: np.ndarray,
) -> PublicTrajectory:
    """Create a PublicTrajectory object."""

    velocities_m_s = estimate_velocity(
        positions_m,
        time_s,
    )

    return PublicTrajectory(
        trajectory_id=trajectory_id,
        time_s=np.asarray(
            time_s,
            dtype=float,
        ),
        positions_m=np.asarray(
            positions_m,
            dtype=float,
        ),
        velocities_m_s=velocities_m_s,
    )


def load_generic_position_csv(
    csv_path: Path,
    time_column: str,
    x_column: str,
    y_column: str,
    z_column: str,
    trajectory_id: str = "public_001",
) -> PublicTrajectory:
    """
    Load a public CSV when it already contains local x/y/z positions.
    """

    dataframe = load_csv(
        csv_path
    )

    time_s = dataframe[
        time_column
    ].to_numpy(
        dtype=float
    )

    positions_m = dataframe[
        [
            x_column,
            y_column,
            z_column,
        ]
    ].to_numpy(
        dtype=float
    )

    return build_public_trajectory(
        trajectory_id=trajectory_id,
        time_s=time_s,
        positions_m=positions_m,
    )


def load_gps_csv(
    csv_path: Path,
    time_column: str,
    latitude_column: str,
    longitude_column: str,
    altitude_column: str,
    trajectory_id: str = "public_gps_001",
) -> PublicTrajectory:
    """
    Load a GPS-based UAV trajectory.

    Latitude and longitude are converted into local metres.
    """

    dataframe = load_csv(
        csv_path
    )

    time_s = dataframe[
        time_column
    ].to_numpy(
        dtype=float
    )

    latitude_deg = dataframe[
        latitude_column
    ].to_numpy(
        dtype=float
    )

    longitude_deg = dataframe[
        longitude_column
    ].to_numpy(
        dtype=float
    )

    altitude_m = dataframe[
        altitude_column
    ].to_numpy(
        dtype=float
    )

    x_m, y_m = (
        latitude_longitude_to_local_xy(
            latitude_deg,
            longitude_deg,
        )
    )

    positions_m = np.column_stack(
        (
            x_m,
            y_m,
            altitude_m,
        )
    )

    return build_public_trajectory(
        trajectory_id=trajectory_id,
        time_s=time_s,
        positions_m=positions_m,
    )


def load_scenario23(
    csv_path: Path,
) -> pd.DataFrame:
    """
    Load DeepSense6G Scenario 23.

    The Scenario 23 CSV acts as a manifest connecting
    each sample to its RGB image, 60 GHz mmWave power
    file, timestamp, and ground-truth beam index.
    """

    dataframe = load_csv(
        csv_path
    )

    required_columns = [
        "index",
        "unit1_rgb",
        "unit1_pwr_60ghz",
        "time_stamp[UTC]",
        "unit1_beam_index",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing Scenario 23 columns: {missing_columns}"
        )

    return dataframe[
        required_columns
    ].copy()

