"""Wireless channel and RIS calculations."""

from dataclasses import replace
from functools import lru_cache
from math import erfc

import numpy as np
from scipy.linalg import hadamard

from atmosphere import atmospheric_loss_linear, specific_gaseous_attenuation_db_per_km
from config import SimulationConfig
from hardware import ris_reflection_coefficients, s11_power_factor, transmitter_factors
from ris import ris_element_positions, ris_normal


def free_space_path_loss(
    distance_m: float,
    config: SimulationConfig,
) -> float:
    """Calculate free-space path loss as a linear power ratio."""

    if np.any(np.asarray(distance_m) <= 0):
        raise ValueError(
            "Distance must be greater than zero."
        )

    return (
        4.0
        * np.pi
        * distance_m
        / config.wavelength_m
    ) ** 2


@lru_cache(maxsize=64)
def _specific_attenuation_db_per_m(
    frequency_hz: float,
    pressure_hpa: float,
    temperature_k: float,
    water_vapour_density_g_m3: float,
) -> float:
    """Cache the ITU-R P.676 line-by-line sum; it does not depend on distance."""

    return specific_gaseous_attenuation_db_per_km(
        frequency_hz,
        pressure_hpa,
        temperature_k,
        water_vapour_density_g_m3,
    ) / 1000.0


def atmospheric_amplitude(
    distance_m: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Return the field-amplitude factor of gaseous absorption over distance."""

    gamma_db_per_m = _specific_attenuation_db_per_m(
        config.carrier_frequency_hz,
        config.atmospheric_pressure_hpa,
        config.atmospheric_temperature_k,
        config.water_vapour_density_g_m3,
    )

    return 10.0 ** (-gamma_db_per_m * np.asarray(distance_m) / 20.0)


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
    distance_m: float | np.ndarray,
    config: SimulationConfig,
) -> complex | np.ndarray:
    """
    Calculate the complex free-space channel coefficient.

    The magnitude represents propagation attenuation.
    The complex phase represents propagation delay.
    Accepts a scalar or an array of distances.
    """

    distance_m = np.asarray(distance_m, dtype=float)

    if np.any(distance_m <= 0):
        raise ValueError(
            "Distance must be greater than zero."
        )

    wavelength_m = config.wavelength_m

    amplitude = wavelength_m / (4.0 * np.pi * distance_m)
    phase = -2.0 * np.pi * distance_m / wavelength_m

    return amplitude * np.exp(1j * phase)


def element_pattern_amplitude(
    directions_m: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """
    Return the field-amplitude pattern of each RIS element.

    directions_m points from each element towards the far node. The power
    pattern is G_e * cos^q(theta) in front of the panel and zero behind it.
    """

    amplitude = np.ones(len(directions_m))

    if config.use_element_gain:
        amplitude *= np.sqrt(np.pi)

    if config.use_obliquity:
        cos_theta = (
            directions_m @ ris_normal(config)
        ) / np.linalg.norm(directions_m, axis=1)
        amplitude *= np.sqrt(np.clip(cos_theta, 0.0, None) ** config.obliquity_exponent)

    return amplitude


def ris_element_channels(
    uav_position_m: np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate BS-to-RIS and RIS-to-UAV complex channels.

    Each hop includes free-space propagation, gaseous absorption, the RIS
    element pattern, and the transceiver antenna gain on its side.

    Returns:
        bs_to_ris: Complex channel for every RIS element.
        ris_to_uav: Complex channel for every RIS element.
    """

    bs_position_m = np.array(config.bs_position_m, dtype=float)
    uav_position_m = np.array(uav_position_m, dtype=float)

    element_positions_m = ris_element_positions(config)

    bs_vectors_m = bs_position_m - element_positions_m
    uav_vectors_m = uav_position_m - element_positions_m

    bs_to_ris_distances_m = np.linalg.norm(bs_vectors_m, axis=1)
    ris_to_uav_distances_m = np.linalg.norm(uav_vectors_m, axis=1)

    bs_gain_amplitude = 10.0 ** (config.bs_antenna_gain_dbi / 20.0)
    uav_gain_amplitude = 10.0 ** (config.uav_antenna_gain_dbi / 20.0)

    bs_to_ris = (
        bs_gain_amplitude
        * complex_free_space_channel(bs_to_ris_distances_m, config)
        * atmospheric_amplitude(bs_to_ris_distances_m, config)
        * element_pattern_amplitude(bs_vectors_m, config)
    )

    ris_to_uav = (
        uav_gain_amplitude
        * complex_free_space_channel(ris_to_uav_distances_m, config)
        * atmospheric_amplitude(ris_to_uav_distances_m, config)
        * element_pattern_amplitude(uav_vectors_m, config)
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


def beamforming_gain_ratio(
    applied_phases_rad: np.ndarray,
    bs_to_ris: np.ndarray,
    ris_to_uav: np.ndarray,
) -> float:
    """
    Return |sum g_n e^{j phi_n}|^2 / (sum |g_n|)^2, in [0, 1].

    This is the fraction of the ideal coherent array gain delivered by a
    phase profile. A common phase offset on every element does not change
    it, unlike the mean absolute phase error.
    """

    cascaded = bs_to_ris * ris_to_uav
    combined = np.sum(cascaded * np.exp(1j * np.asarray(applied_phases_rad)))
    ideal = np.sum(np.abs(cascaded))

    if ideal == 0:
        return 0.0

    return float(np.abs(combined) ** 2 / ideal**2)


def ris_complex_channel(
    uav_position_m: np.ndarray,
    config: SimulationConfig,
    commanded_phases_rad: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> complex:
    """
    Calculate the end-to-end RIS-assisted complex channel.

    Each RIS element contributes:

        h_BS,RIS[n] * Gamma[n] * h_RIS,UAV[n]

    where Gamma[n] is the reflection coefficient the hardware realises
    for the commanded phase. If no phases are given the optimal ones for
    the true position are used (genie-aided control).
    """

    bs_to_ris, ris_to_uav = ris_element_channels(
        uav_position_m,
        config,
    )

    if commanded_phases_rad is None:
        commanded_phases_rad = optimal_ris_phases(bs_to_ris, ris_to_uav)

    reflection_coefficients = ris_reflection_coefficients(
        commanded_phases_rad,
        config,
        rng,
    )

    return np.sum(
        bs_to_ris
        * reflection_coefficients
        * ris_to_uav
    )


def link_metrics(
    uav_position_m: np.ndarray,
    config: SimulationConfig,
    commanded_phases_rad: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Return received power, SNR, SNDR, spectral efficiency and BER.

    SNDR includes PA distortion, which scales with the received signal:

        SNDR = rho_u * P|h|^2 / (rho_d * P|h|^2 + N0)
    """

    channel = ris_complex_channel(
        uav_position_m,
        config,
        commanded_phases_rad,
        rng,
    )

    useful_fraction, distortion_fraction = transmitter_factors(config)
    port_factor = s11_power_factor(config) ** 2

    total_received_w = (
        config.transmit_power_w
        * port_factor
        * np.abs(channel) ** 2
    )

    signal_w = useful_fraction * total_received_w
    distortion_w = distortion_fraction * total_received_w
    noise_w = config.noise_power_w

    snr = signal_w / noise_w
    sndr = signal_w / (distortion_w + noise_w)

    return {
        "channel": channel,
        "received_power_w": total_received_w,
        "snr_linear": snr,
        "snr_db": linear_to_db(snr),
        "sndr_linear": sndr,
        "sndr_db": linear_to_db(sndr),
        "spectral_efficiency_bps_hz": float(np.log2(1.0 + sndr)),
        "ber_qpsk": ber_mqam(sndr, 4),
        "ber_16qam": ber_mqam(sndr, 16),
    }


def physical_ris_received_power_w(
    uav_position_m: np.ndarray,
    config: SimulationConfig,
) -> float:
    """
    Calculate received power using the physical RIS channel.

    Atmospheric attenuation is applied separately to the
    BS-to-RIS and RIS-to-UAV paths.
    """

    return link_metrics(
        uav_position_m,
        config,
    )["received_power_w"]


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


def linear_to_db(
    value: float,
) -> float:
    """Convert a linear power ratio to dB, flooring at -300 dB."""

    return float(10.0 * np.log10(max(value, 1e-30)))


def ber_mqam(
    snr_linear: float,
    modulation_order: int,
) -> float:
    """
    Approximate BER of Gray-coded square M-QAM in AWGN.

    Uses the nearest-neighbour approximation; exact for QPSK.
    """

    bits = np.log2(modulation_order)
    root_m = np.sqrt(modulation_order)

    symbol_term = erfc(np.sqrt(3.0 * snr_linear / (2.0 * (modulation_order - 1))))

    return float(min(0.5, 2.0 * (1.0 - 1.0 / root_m) / bits * symbol_term))


def estimate_cascaded_channel(
    uav_position_m: np.ndarray,
    config: SimulationConfig,
    pilot_symbols_per_pattern: int = 1024,
    hardware_aware: bool = False,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Least-squares estimate of the per-element cascaded channel.

    The RIS cycles through N Hadamard (0/pi) training patterns and holds
    each for a known pilot sequence; correlating over the sequence gives a
    processing gain equal to its length. Without it the per-element SNR is
    the coherent link SNR divided by N, which is far below 0 dB for a large
    RIS. The naive estimator assumes the RIS realised the commanded
    patterns exactly; the hardware-aware estimator uses the mean reflection
    coefficient the hardware model predicts for each commanded state,
    which removes the bias from amplitude-phase coupling. Phase noise
    remains a random error for both.

    Returns NMSE (linear and dB) of the estimate.
    """

    if rng is None:
        rng = np.random.default_rng(config.impairment_seed)

    bs_to_ris, ris_to_uav = ris_element_channels(uav_position_m, config)
    cascaded = bs_to_ris * ris_to_uav
    elements = len(cascaded)

    if elements & (elements - 1):
        raise ValueError("Hadamard training needs a power-of-two element count.")

    patterns = hadamard(elements).astype(float)
    base_phase = training_base_phase(config) if hardware_aware else 0.0
    commanded_phases = np.where(patterns > 0, base_phase, base_phase + np.pi)

    realised = np.stack([
        ris_reflection_coefficients(row, config, rng)
        for row in commanded_phases
    ])

    useful_fraction, distortion_fraction = transmitter_factors(config)
    amplitude = np.sqrt(
        config.transmit_power_w
        * pilot_symbols_per_pattern
        * useful_fraction
        * s11_power_factor(config) ** 2
    )

    received_signal = amplitude * realised @ cascaded
    noise_variance = (
        config.noise_power_w
        + distortion_fraction * np.abs(received_signal) ** 2 / useful_fraction
    )
    received = received_signal + np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(elements) + 1j * rng.standard_normal(elements)
    )

    if hardware_aware:
        # LMMSE with an i.i.d. prior of the mean per-element channel power,
        # which in practice comes from long-term statistics or geometry.
        # PA distortion is signal-dependent, so only its average power is
        # usable as a noise model (as a real receiver would know it).
        training = amplitude * np.stack([
            ris_reflection_coefficients(row, _without_noise(config), rng)
            for row in commanded_phases
        ])
        prior_power = np.mean(np.abs(cascaded) ** 2)
        gram = (
            prior_power * training @ training.conj().T
            + np.mean(noise_variance) * np.eye(elements)
        )
        estimate = prior_power * training.conj().T @ np.linalg.solve(gram, received)
    else:
        nominal = np.sqrt(config.ris_reflection_efficiency) * patterns
        estimate = np.linalg.solve(amplitude * nominal, received)

    nmse = float(np.sum(np.abs(estimate - cascaded) ** 2) / np.sum(np.abs(cascaded) ** 2))

    return {
        "nmse_linear": nmse,
        "nmse_db": linear_to_db(nmse),
        "estimate": estimate,
        "cascaded": cascaded,
    }


def training_base_phase(
    config: SimulationConfig,
) -> float:
    """
    Return the phase p such that training on (p, p + pi) reflects most energy.

    With amplitude-phase coupling the two states of an antipodal pair have
    amplitudes a and b. LMMSE copes with the imbalance (a != b), so the
    pair with the largest a + b gives the best pilot SNR.
    """

    bits = config.ris_phase_bits if config.use_ris_phase_quantization else 6
    candidates = np.arange(2 ** (bits - 1)) * 2.0 * np.pi / 2**bits

    deterministic = _without_noise(config)
    first = np.abs(ris_reflection_coefficients(candidates, deterministic))
    second = np.abs(ris_reflection_coefficients(candidates + np.pi, deterministic))

    return float(candidates[np.argmax(first + second)])


def _without_noise(
    config: SimulationConfig,
) -> SimulationConfig:
    """Return the config with only the deterministic RIS impairments."""

    return replace(config, use_ris_phase_noise=False)
