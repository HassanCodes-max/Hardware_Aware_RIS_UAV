"""Generate a RIS phase-tracking dataset for the moving UAV."""

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
from trajectory import straight_line_trajectory


def run_phase_tracking(output_path: Path) -> None:
    """Generate RIS phase-tracking data for the UAV trajectory."""

    config = SimulationConfig()

    trajectory = straight_line_trajectory(config)

    phase_columns = [
        f"phase_{index}_rad"
        for index in range(1, config.ris_elements + 1)
    ]

    phase_sin_columns = [
        f"phase_{index}_sin"
        for index in range(1, config.ris_elements + 1)
    ]

    phase_cos_columns = [
        f"phase_{index}_cos"
        for index in range(1, config.ris_elements + 1)
    ]

    header = [
        "time_s",
        "uav_x_m",
        "uav_y_m",
        "uav_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "channel_magnitude",
        "channel_phase_rad",
        "received_power_w",
        "snr_linear",
        "snr_db",
        "max_phase_change_rad",
        "mean_phase_change_rad",
    ]

    header.extend(phase_columns)
    header.extend(phase_sin_columns)
    header.extend(phase_cos_columns)

    previous_phases = None

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.writer(output_file)

        writer.writerow(header)

        for time_s, position_m, velocity_m_s in zip(
            trajectory.time_s,
            trajectory.positions_m,
            trajectory.velocities_m_s,
        ):

            uav_position_m = np.array(
                position_m,
                dtype=float,
            )

            velocity_m_s = np.array(
                velocity_m_s,
                dtype=float,
            )

            bs_to_ris, ris_to_uav = (
                ris_element_channels(
                    uav_position_m,
                    config,
                )
            )

            phases = optimal_ris_phases(
                bs_to_ris,
                ris_to_uav,
            )

            channel = ris_complex_channel(
                uav_position_m,
                config,
            )

            received_power = (
                physical_ris_received_power_w(
                    uav_position_m,
                    config,
                )
            )

            snr_linear = (
                received_power
                / config.noise_power_w
            )

            snr_db = (
                10.0
                * np.log10(snr_linear)
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
                    np.abs(phase_difference)
                )

                max_phase_change = np.max(
                    absolute_phase_difference
                )

                mean_phase_change = np.mean(
                    absolute_phase_difference
                )

            row = [
                f"{time_s:.4f}",
                f"{uav_position_m[0]:.6f}",
                f"{uav_position_m[1]:.6f}",
                f"{uav_position_m[2]:.6f}",
                f"{velocity_m_s[0]:.6f}",
                f"{velocity_m_s[1]:.6f}",
                f"{velocity_m_s[2]:.6f}",
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

            writer.writerow(row)

            previous_phases = phases.copy()


if __name__ == "__main__":

    result_file = (
        Path(__file__).parent
        / "results"
        / "phase_tracking.csv"
    )

    run_phase_tracking(result_file)

    print(
        f"Phase-tracking dataset written to {result_file}"
    )