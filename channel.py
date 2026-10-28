"""Wireless channel and RIS calculations."""

import numpy as np

from atmosphere import atmospheric_loss_db, atmospheric_loss_linear
from config import SimulationConfig
from ris import ris_element_positions


def free_space_path_loss(
    distance_m: float,
    config: SimulationConfig,
) -> float:
    """Calculate free-space path loss as a linear power ratio."""

    if distance_m <= 0:
        raise ValueError(
            "Distance must be greater than zero."
        )

    wavelength_m = config.wavelength_m

    return (
        4.0
        * np.pi
        * distance_m
        / wavelength_m
    ) ** 2


def received_power_w(
    distance_m: float,
    config: SimulationConfig,
) -> float:
    """Calculate received power including atmospheric loss."""

    path_loss = free_space_path_loss(
        distance_m,
        config,
    )

    atmospheric_loss = atmospheric_loss_linear(
        distance_m,
        config.carrier_frequency_hz,
        config.atmospheric_pressure_hpa,
        config.atmospheric_temperature_k,
        config.water_vapour_density_g_m3,
    )

    return (
        config.transmit_power_w
        / (
            path_loss
            * atmospheric_loss
        )
    )


def complex_free_space_channel(
    distance_m: float,
    config: SimulationConfig,
) -> complex:
    """
    Calculate the complex free-space channel coefficient.

    The magnitude represents propagation attenuation.
    The complex phase represents propagation delay.
    """

    if distance_m <= 0:
        raise ValueError(
            "Distance must be greater than zero."
        )

    wavelength_m = config.wavelength_m

    amplitude = (
        wavelength_m
        / (
            4.0
            * np.pi
            * distance_m
        )
    )

    phase = (
        -2.0
        * np.pi
        * distance_m
        / wavelength_m
    )

    return amplitude * np.exp(1j * phase)


def ris_element_channels(
    uav_position_m: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate BS-to-RIS and RIS-to-UAV complex channels.

    Returns:
        bs_to_ris: Complex channel for every RIS element.
        ris_to_uav: Complex channel for every RIS element.
    """

    bs_position_m = np.array(
        config.bs_position_m,
        dtype=float,
    )

    uav_position_m = np.array(
        uav_position_m,
        dtype=float,
    )

    element_positions_m = ris_element_positions(
        config
    )

    bs_to_ris_distances_m = np.linalg.norm(
        element_positions_m - bs_position_m,
        axis=1,
    )

    ris_to_uav_distances_m = np.linalg.norm(
        element_positions_m - uav_position_m,
        axis=1,
    )

    bs_to_ris = np.array(
        [
            complex_free_space_channel(
                distance,
                config,
            )
            for distance in bs_to_ris_distances_m
        ]
    )

    ris_to_uav = np.array(
        [
            complex_free_space_channel(
                distance,
                config,
            )
            for distance in ris_to_uav_distances_m
        ]
    )

    return bs_to_ris, ris_to_uav


def optimal_ris_phases(
    bs_to_ris: np.ndarray,
    ris_to_uav: np.ndarray,
) -> np.ndarray:
    """
    Calculate ideal RIS phases for coherent combining.

    The phase of each RIS element is selected so that
    all reflected contributions add constructively.
    """

    cascaded_channel = (
        bs_to_ris
        * ris_to_uav
    )

    return -np.angle(
        cascaded_channel
    )


def ris_complex_channel(
    uav_position_m: np.ndarray,
    config: SimulationConfig,
) -> complex:
    """
    Calculate the optimized RIS-assisted complex channel.

    Each RIS element contributes:

        h_BS,RIS[n] * Gamma[n] * h_RIS,UAV[n]

    where Gamma[n] is the RIS reflection coefficient.
    """

    bs_to_ris, ris_to_uav = ris_element_channels(
        uav_position_m,
        config,
    )

    phases = optimal_ris_phases(
        bs_to_ris,
        ris_to_uav,
    )

    reflection_amplitude = np.sqrt(
        config.ris_reflection_efficiency
    )

    reflection_coefficients = (
        reflection_amplitude
        * np.exp(1j * phases)
    )

    element_contributions = (
        bs_to_ris
        * reflection_coefficients
        * ris_to_uav
    )

    return np.sum(
        element_contributions
    )


def physical_ris_received_power_w(
    uav_position_m: np.ndarray,
    config: SimulationConfig,
) -> float:
    """
    Calculate received power using the physical RIS channel.

    Atmospheric attenuation is applied separately to the
    BS-to-RIS and RIS-to-UAV paths.
    """

    bs_to_ris, ris_to_uav = ris_element_channels(
        uav_position_m,
        config,
    )

    phases = optimal_ris_phases(
        bs_to_ris,
        ris_to_uav,
    )

    reflection_amplitude = np.sqrt(
        config.ris_reflection_efficiency
    )

    reflection_coefficients = (
        reflection_amplitude
        * np.exp(1j * phases)
    )

    element_positions_m = ris_element_positions(
        config
    )

    bs_position_m = np.array(
        config.bs_position_m,
        dtype=float,
    )

    uav_position_m = np.array(
        uav_position_m,
        dtype=float,
    )

    bs_to_ris_distances_m = np.linalg.norm(
        element_positions_m - bs_position_m,
        axis=1,
    )

    ris_to_uav_distances_m = np.linalg.norm(
        element_positions_m - uav_position_m,
        axis=1,
    )

    atmospheric_amplitude_bs_ris = np.array(
        [
            10.0 ** (
                -atmospheric_loss_db(
                    distance,
                    config.carrier_frequency_hz,
                    config.atmospheric_pressure_hpa,
                    config.atmospheric_temperature_k,
                    config.water_vapour_density_g_m3,
                )
                / 20.0
            )
            for distance in bs_to_ris_distances_m
        ]
    )

    atmospheric_amplitude_ris_uav = np.array(
        [
            10.0 ** (
                -atmospheric_loss_db(
                    distance,
                    config.carrier_frequency_hz,
                    config.atmospheric_pressure_hpa,
                    config.atmospheric_temperature_k,
                    config.water_vapour_density_g_m3,
                )
                / 20.0
            )
            for distance in ris_to_uav_distances_m
        ]
    )

    element_contributions = (
        bs_to_ris
        * atmospheric_amplitude_bs_ris
        * reflection_coefficients
        * ris_to_uav
        * atmospheric_amplitude_ris_uav
    )

    total_channel = np.sum(
        element_contributions
    )

    return (
        config.transmit_power_w
        * np.abs(total_channel) ** 2
    )


def ris_received_power_w(
    uav_position_m: np.ndarray,
    user_position_m: np.ndarray,
    ris_position_m: np.ndarray,
    config: SimulationConfig,
) -> float:
    """
    Calculate RIS-assisted received power.

    This function is retained as the previous simplified
    model for comparison with the physical RIS model.
    """

    user_to_ris_distance_m = np.linalg.norm(
        user_position_m
        - ris_position_m
    )

    ris_to_uav_distance_m = np.linalg.norm(
        ris_position_m
        - uav_position_m
    )

    path_loss_user_ris = free_space_path_loss(
        user_to_ris_distance_m,
        config,
    )

    path_loss_ris_uav = free_space_path_loss(
        ris_to_uav_distance_m,
        config,
    )

    atmospheric_loss_user_ris = atmospheric_loss_linear(
        user_to_ris_distance_m,
        config.carrier_frequency_hz,
        config.atmospheric_pressure_hpa,
        config.atmospheric_temperature_k,
        config.water_vapour_density_g_m3,
    )

    atmospheric_loss_ris_uav = atmospheric_loss_linear(
        ris_to_uav_distance_m,
        config.carrier_frequency_hz,
        config.atmospheric_pressure_hpa,
        config.atmospheric_temperature_k,
        config.water_vapour_density_g_m3,
    )

    ris_gain = (
        config.ris_elements ** 2
        * config.ris_reflection_efficiency
    )

    total_path_loss = (
        path_loss_user_ris
        * atmospheric_loss_user_ris
        * path_loss_ris_uav
        * atmospheric_loss_ris_uav
    )

    return (
        config.transmit_power_w
        * ris_gain
        / total_path_loss
    )


def spectral_efficiency_bps_hz(
    received_power_w: float,
    config: SimulationConfig,
) -> float:
    """Calculate ideal Shannon spectral efficiency."""

    snr = (
        received_power_w
        / config.noise_power_w
    )

    return np.log2(
        1.0 + snr
    )