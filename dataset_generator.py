"""Generate a diverse RIS beam-tracking dataset."""

import csv
from pathlib import Path

import numpy as np

from channel import (
    optimal_ris_phases,
    physical_ris_received_power_w,
    ris_complex_channel,
    ris_element_channels,
)
from config import SimulationConfig
from trajectory import (
    Trajectory,
    circular_trajectory,
    helical_trajectory,
    random_trajectory,
    straight_line_custom_trajectory,
    waypoint_trajectory,
)


def trajectory_metadata(
    trajectory_id: int,
    trajectory_type: str,
) -> dict:
    """Return metadata for one trajectory."""

    return {
        "trajectory_id": trajectory_id,
        "trajectory_type": trajectory_type,
    }


def generate_designed_trajectories(
    config: SimulationConfig,
) -> list[tuple[int, str, Trajectory]]:
    """Generate manually designed trajectory scenarios."""

    trajectories = []

    trajectory_id = 1

    scenarios = [
        (
            "straight_x",
            straight_line_custom_trajectory(
                initial_position_m=(20.0, 0.0, 20.0),
                speed_m_s=10.0,
                direction=(1.0, 0.0, 0.0),
                duration_s=config.flight_duration_s,
                time_step_s=config.time_step_s,
            ),
        ),
        (
            "straight_negative_x",
            straight_line_custom_trajectory(
                initial_position_m=(100.0, 0.0, 25.0),
                speed_m_s=8.0,
                direction=(-1.0, 0.0, 0.0),
                duration_s=config.flight_duration_s,
                time_step_s=config.time_step_s,
            ),
        ),
        (
            "straight_y",
            straight_line_custom_trajectory(
                initial_position_m=(0.0, -50.0, 30.0),
                speed_m_s=12.0,
                direction=(0.0, 1.0, 0.0),
                duration_s=config.flight_duration_s,
                time_step_s=config.time_step_s,
            ),
        ),
        (
            "diagonal",
            straight_line_custom_trajectory(
                initial_position_m=(-40.0, -40.0, 25.0),
                speed_m_s=12.0,
                direction=(1.0, 1.0, 0.1),
                duration_s=config.flight_duration_s,
                time_step_s=config.time_step_s,
            ),
        ),
        (
            "ascending",
            straight_line_custom_trajectory(
                initial_position_m=(20.0, 0.0, 10.0),
                speed_m_s=8.0,
                direction=(0.2, 0.1, 1.0),
                duration_s=config.flight_duration_s,
                time_step_s=config.time_step_s,
            ),
        ),
        (
            "descending",
            straight_line_custom_trajectory(
                initial_position_m=(100.0, 40.0, 70.0),
                speed_m_s=8.0,
                direction=(-0.2, -0.1, -1.0),
                duration_s=config.flight_duration_s,
                time_step_s=config.time_step_s,
            ),
        ),
        (
            "circle",
            circular_trajectory(
                center_m=(40.0, 0.0, 30.0),
                radius_m=40.0,
                altitude_m=30.0,
                angular_speed_rad_s=10.0 / 40.0,
                duration_s=config.flight_duration_s,
                time_step_s=config.time_step_s,
            ),
        ),
        (
            "helix",
            helical_trajectory(
                center_m=(30.0, 20.0, 20.0),
                radius_m=35.0,
                initial_altitude_m=20.0,
                vertical_speed_m_s=1.0,
                angular_speed_rad_s=10.0 / 35.0,
                duration_s=config.flight_duration_s,
                time_step_s=config.time_step_s,
            ),
        ),
        (
            "waypoints",
            waypoint_trajectory(
                waypoints_m=[
                    (20.0, 0.0, 20.0),
                    (80.0, 20.0, 25.0),
                    (40.0, 80.0, 35.0),
                    (-20.0, 40.0, 30.0),
                    (20.0, 0.0, 20.0),
                ],
                speed_m_s=10.0,
                time_step_s=config.time_step_s,
            ),
        ),
    ]

    for trajectory_type, trajectory in scenarios:

        trajectories.append(
            (
                trajectory_id,
                trajectory_type,
                trajectory,
            )
        )

        trajectory_id += 1

    return trajectories


def generate_random_trajectories(
    config: SimulationConfig,
    number_of_trajectories: int,
    seed: int,
    starting_id: int,
) -> list[tuple[int, str, Trajectory]]:
    """Generate many randomized trajectories."""

    rng = np.random.default_rng(
        seed
    )

    trajectories = []

    for offset in range(
        number_of_trajectories
    ):

        trajectory = random_trajectory(
            rng=rng,
            duration_s=config.flight_duration_s,
            time_step_s=config.time_step_s,
        )

        trajectory_type = (
            "random"
        )

        trajectories.append(
            (
                starting_id + offset,
                trajectory_type,
                trajectory,
            )
        )

    return trajectories


def write_trajectory_dataset(
    trajectories: list[tuple[int, str, Trajectory]],
    output_path: Path,
) -> None:
    """Run trajectories through the RIS simulator and save the dataset."""

    config = SimulationConfig()

    phase_columns = [
        f"phase_{index}_rad"
        for index in range(
            1,
            config.ris_elements + 1,
        )
    ]

    phase_sin_columns = [
        f"phase_{index}_sin"
        for index in range(
            1,
            config.ris_elements + 1,
        )
    ]

    phase_cos_columns = [
        f"phase_{index}_cos"
        for index in range(
            1,
            config.ris_elements + 1,
        )
    ]

    header = [
        "trajectory_id",
        "trajectory_type",
        "time_s",
        "uav_x_m",
        "uav_y_m",
        "uav_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "speed_m_s",
        "channel_magnitude",
        "channel_phase_rad",
        "received_power_w",
        "snr_linear",
        "snr_db",
        "max_phase_change_rad",
        "mean_phase_change_rad",
    ]

    header.extend(
        phase_columns
    )

    header.extend(
        phase_sin_columns
    )

    header.extend(
        phase_cos_columns
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.writer(
            output_file
        )

        writer.writerow(
            header
        )

        for (
            trajectory_id,
            trajectory_type,
            trajectory,
        ) in trajectories:

            previous_phases = None

            for (
                time_s,
                position_m,
                velocity_m_s,
            ) in zip(
                trajectory.time_s,
                trajectory.positions_m,
                trajectory.velocities_m_s,
            ):

                position_m = np.asarray(
                    position_m,
                    dtype=float,
                )

                velocity_m_s = np.asarray(
                    velocity_m_s,
                    dtype=float,
                )

                speed_m_s = np.linalg.norm(
                    velocity_m_s
                )

                bs_to_ris, ris_to_uav = (
                    ris_element_channels(
                        position_m,
                        config,
                    )
                )

                phases = (
                    optimal_ris_phases(
                        bs_to_ris,
                        ris_to_uav,
                    )
                )

                channel = (
                    ris_complex_channel(
                        position_m,
                        config,
                    )
                )

                received_power = (
                    physical_ris_received_power_w(
                        position_m,
                        config,
                    )
                )

                snr_linear = (
                    received_power
                    / config.noise_power_w
                )

                snr_db = (
                    10.0
                    * np.log10(
                        max(
                            snr_linear,
                            1e-30,
                        )
                    )
                )

                if previous_phases is None:

                    max_phase_change = 0.0
                    mean_phase_change = 0.0

                else:

                    phase_difference = np.angle(
                        np.exp(
                            1j
                            * (
                                phases
                                - previous_phases
                            )
                        )
                    )

                    absolute_phase_difference = (
                        np.abs(
                            phase_difference
                        )
                    )

                    max_phase_change = (
                        np.max(
                            absolute_phase_difference
                        )
                    )

                    mean_phase_change = (
                        np.mean(
                            absolute_phase_difference
                        )
                    )

                row = [
                    trajectory_id,
                    trajectory_type,
                    f"{time_s:.4f}",
                    f"{position_m[0]:.6f}",
                    f"{position_m[1]:.6f}",
                    f"{position_m[2]:.6f}",
                    f"{velocity_m_s[0]:.6f}",
                    f"{velocity_m_s[1]:.6f}",
                    f"{velocity_m_s[2]:.6f}",
                    f"{speed_m_s:.6f}",
                    f"{abs(channel):.12e}",
                    f"{np.angle(channel):.6f}",
                    f"{received_power:.12e}",
                    f"{snr_linear:.12e}",
                    f"{snr_db:.6f}",
                    f"{max_phase_change:.6f}",
                    f"{mean_phase_change:.6f}",
                ]

                row.extend(
                    f"{phase:.6f}"
                    for phase in phases
                )

                row.extend(
                    f"{np.sin(phase):.6f}"
                    for phase in phases
                )

                row.extend(
                    f"{np.cos(phase):.6f}"
                    for phase in phases
                )

                writer.writerow(
                    row
                )

                previous_phases = (
                    phases.copy()
                )


def main() -> None:
    """Generate the complete synthetic RIS dataset."""

    config = SimulationConfig()

    designed_trajectories = (
        generate_designed_trajectories(
            config
        )
    )

    random_trajectories = (
        generate_random_trajectories(
            config=config,
            number_of_trajectories=91,
            seed=42,
            starting_id=10,
        )
    )

    all_trajectories = (
        designed_trajectories
        + random_trajectories
    )

    output_path = (
        Path(__file__).parent
        / "results"
        / "diverse_ris_dataset.csv"
    )

    write_trajectory_dataset(
        trajectories=all_trajectories,
        output_path=output_path,
    )

    print(
        "Dataset generation complete."
    )

    print(
        f"Trajectories: "
        f"{len(all_trajectories)}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()