"""Run the sub-THz UAV communication simulation."""

import csv
from pathlib import Path

import numpy as np

from atmosphere import atmospheric_loss_db
from channel import (
    free_space_path_loss,
    physical_ris_received_power_w,
    ris_received_power_w,
    spectral_efficiency_bps_hz,
)
from config import SimulationConfig
from ris import ris_element_positions
from trajectory import straight_line_trajectory


def run_simulation(output_path: Path) -> None:
    """Simulate UAV movement with both RIS models."""

    config = SimulationConfig()

    trajectory = straight_line_trajectory(config)

    user_position_m = np.array(
        [0.0, 0.0, 0.0]
    )

    ris_position_m = np.array(
        config.ris_position_m,
        dtype=float,
    )

    ris_positions_m = ris_element_positions(
        config
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as output_file:

        writer = csv.writer(output_file)

        writer.writerow(
            [
                "time_s",
                "uav_x_m",
                "uav_y_m",
                "uav_z_m",
                "uav_user_distance_m",
                "user_ris_distance_m",
                "ris_uav_center_distance_m",
                "ris_uav_min_element_distance_m",
                "ris_uav_max_element_distance_m",
                "fspl_user_ris_db",
                "fspl_ris_uav_center_db",
                "atmospheric_loss_user_ris_db",
                "atmospheric_loss_ris_uav_center_db",
                "simplified_ris_gain_db",
                "simplified_received_power_w",
                "physical_received_power_w",
                "physical_snr_linear",
                "physical_snr_db",
                "physical_spectral_efficiency_bps_hz",
            ]
        )

        for time_s, position_m in zip(
            trajectory.time_s,
            trajectory.positions_m
        ):

            uav_position_m = np.array(
                position_m,
                dtype=float,
            )

            uav_user_distance_m = np.linalg.norm(
                uav_position_m
                - user_position_m
            )

            user_ris_distance_m = np.linalg.norm(
                user_position_m
                - ris_position_m
            )

            ris_uav_center_distance_m = np.linalg.norm(
                uav_position_m
                - ris_position_m
            )

            ris_uav_element_distances_m = np.linalg.norm(
                ris_positions_m
                - uav_position_m,
                axis=1,
            )

            ris_uav_min_element_distance_m = np.min(
                ris_uav_element_distances_m
            )

            ris_uav_max_element_distance_m = np.max(
                ris_uav_element_distances_m
            )

            fspl_user_ris_linear = free_space_path_loss(
                user_ris_distance_m,
                config
            )

            fspl_ris_uav_center_linear = free_space_path_loss(
                ris_uav_center_distance_m,
                config
            )

            fspl_user_ris_db = 10.0 * np.log10(
                fspl_user_ris_linear
            )

            fspl_ris_uav_center_db = 10.0 * np.log10(
                fspl_ris_uav_center_linear
            )

            atmospheric_loss_user_ris_db = atmospheric_loss_db(
                user_ris_distance_m,
                config.carrier_frequency_hz,
                config.atmospheric_pressure_hpa,
                config.atmospheric_temperature_k,
                config.water_vapour_density_g_m3
            )

            atmospheric_loss_ris_uav_center_db = atmospheric_loss_db(
                ris_uav_center_distance_m,
                config.carrier_frequency_hz,
                config.atmospheric_pressure_hpa,
                config.atmospheric_temperature_k,
                config.water_vapour_density_g_m3
            )

            ris_gain_linear = (
                config.ris_elements ** 2
                * config.ris_reflection_efficiency
            )

            simplified_ris_gain_db = 10.0 * np.log10(
                ris_gain_linear
            )

            simplified_received_power = (
                ris_received_power_w(
                    uav_position_m,
                    user_position_m,
                    ris_position_m,
                    config
                )
            )

            physical_received_power = (
                physical_ris_received_power_w(
                    uav_position_m,
                    config
                )
            )

            physical_snr_linear = (
                physical_received_power
                / config.noise_power_w
            )

            physical_snr_db = 10.0 * np.log10(
                physical_snr_linear
            )

            physical_spectral_efficiency = (
                spectral_efficiency_bps_hz(
                    physical_received_power,
                    config
                )
            )

            writer.writerow(
                [
                    f"{time_s:.2f}",
                    f"{uav_position_m[0]:.6f}",
                    f"{uav_position_m[1]:.6f}",
                    f"{uav_position_m[2]:.6f}",
                    f"{uav_user_distance_m:.6f}",
                    f"{user_ris_distance_m:.6f}",
                    f"{ris_uav_center_distance_m:.6f}",
                    f"{ris_uav_min_element_distance_m:.6f}",
                    f"{ris_uav_max_element_distance_m:.6f}",
                    f"{fspl_user_ris_db:.6f}",
                    f"{fspl_ris_uav_center_db:.6f}",
                    f"{atmospheric_loss_user_ris_db:.6f}",
                    f"{atmospheric_loss_ris_uav_center_db:.6f}",
                    f"{simplified_ris_gain_db:.6f}",
                    f"{simplified_received_power:.12e}",
                    f"{physical_received_power:.12e}",
                    f"{physical_snr_linear:.12e}",
                    f"{physical_snr_db:.6f}",
                    f"{physical_spectral_efficiency:.6f}",
                ]
            )


if __name__ == "__main__":

    result_file = (
        Path(__file__).parent
        / "results"
        / "simulation.csv"
    )

    run_simulation(result_file)

    print(
        f"Simulation results written to {result_file}"
    )