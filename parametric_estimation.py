"""Position-parametric RIS channel estimation.

The BS and RIS are fixed, so the cascaded channel g_n = h1_n * h2_n is a
known function of the UAV position p (3 unknowns), not N free complex
coefficients. Starting from a coarse position prior (e.g. consumer GPS),
we send M << N random training patterns, then refine p by maximising the
likelihood of the pilots under an unknown complex path gain:

    p_hat = argmax_p |y^H A g(p)|^2 / ||A g(p)||^2

with a coarse-to-fine search over the UAV direction around the prior. The RIS is then
configured from g(p_hat) with the hardware-aware phase search.
"""

import numpy as np

from channel import (
    _without_noise,
    atmospheric_amplitude,
    complex_free_space_channel,
    element_pattern_amplitude,
    ris_element_channels,
)
from config import SimulationConfig
from hardware import ris_reflection_coefficients, s11_power_factor, transmitter_factors
from ris import ris_element_positions, ris_normal, ris_panel_axes


def cascaded_channel(
    position_m: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Return g(p) = h1 * h2 for a hypothesised UAV position."""

    bs_to_ris, ris_to_uav = ris_element_channels(position_m, config)

    return bs_to_ris * ris_to_uav


def random_training(
    patterns: int,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return commanded phases (patterns, N) drawn from the realisable levels."""

    levels = 2**config.ris_phase_bits if config.use_ris_phase_quantization else 64

    return rng.integers(0, levels, size=(patterns, config.ris_elements)) * 2.0 * np.pi / levels


def send_pilots(
    true_position_m: np.ndarray,
    commanded_phases: np.ndarray,
    config: SimulationConfig,
    pilot_symbols_per_pattern: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (received pilots, nominal training matrix).

    The received signal uses the RIS coefficients the hardware actually
    realises (with phase noise); the nominal matrix is what the receiver
    can predict from the hardware model.
    """

    useful_fraction, distortion_fraction = transmitter_factors(config)
    amplitude = np.sqrt(
        config.transmit_power_w
        * pilot_symbols_per_pattern
        * useful_fraction
        * s11_power_factor(config) ** 2
    )

    realised = np.stack([ris_reflection_coefficients(row, config, rng) for row in commanded_phases])
    nominal = amplitude * np.stack([
        ris_reflection_coefficients(row, _without_noise(config)) for row in commanded_phases
    ])

    signal = amplitude * realised @ cascaded_channel(true_position_m, config)
    noise_variance = config.noise_power_w + distortion_fraction / useful_fraction * np.mean(np.abs(signal) ** 2)
    received = signal + np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )

    return received, nominal


def cascaded_channels_batch(
    positions_m: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Return g(p) for many hypothesised positions at once, shape (C, N)."""

    bs_to_ris, _ = ris_element_channels(positions_m[0], config)
    elements_m = ris_element_positions(config)

    vectors_m = positions_m[:, None, :] - elements_m[None, :, :]
    distances_m = np.linalg.norm(vectors_m, axis=2)

    ris_to_uav = (
        10.0 ** (config.uav_antenna_gain_dbi / 20.0)
        * complex_free_space_channel(distances_m, config)
        * atmospheric_amplitude(distances_m, config)
        * element_pattern_amplitude(vectors_m.reshape(-1, 3), config).reshape(distances_m.shape)
    )

    return bs_to_ris[None, :] * ris_to_uav


def refine_position(
    prior_m: np.ndarray,
    received: np.ndarray,
    nominal: np.ndarray,
    config: SimulationConfig,
    search_half_width: float = 0.5,
    coarse_step: float = 0.02,
    fine_step: float = 0.002,
) -> np.ndarray:
    """
    Search the UAV direction seen from the RIS, keeping the prior range.

    Directions are direction cosines (u, v) along the panel's column and
    row axes. The UAV is in the RIS far field (beyond about 2D^2/lambda),
    so the pilots constrain direction sharply and range only weakly.
    A coarse grid finer than the beam width (about 2/side) avoids
    locking onto sidelobes; a fine grid then polishes the peak.
    """

    centre_m = np.asarray(config.ris_position_m, dtype=float)
    normal = ris_normal(config)
    column_axis, row_axis = ris_panel_axes(config)

    offset = np.asarray(prior_m, dtype=float) - centre_m
    range_m = np.linalg.norm(offset)
    u0, v0 = offset @ column_axis / range_m, offset @ row_axis / range_m

    def best_direction(u_centre, v_centre, half_width, step):
        values = np.arange(-half_width, half_width + step / 2, step)
        u, v = np.meshgrid(u_centre + values, v_centre + values, indexing="ij")
        u, v = u.ravel(), v.ravel()
        inside = u**2 + v**2 < 1.0
        u, v = u[inside], v[inside]
        w = np.sqrt(1.0 - u**2 - v**2)

        directions = u[:, None] * column_axis + v[:, None] * row_axis + w[:, None] * normal
        candidates = centre_m + range_m * directions

        predicted = cascaded_channels_batch(candidates, config) @ nominal.T
        scores = np.abs(predicted.conj() @ received) ** 2 / np.sum(np.abs(predicted) ** 2, axis=1)
        best = np.argmax(scores)

        return u[best], v[best], candidates[best]

    u, v, _ = best_direction(u0, v0, search_half_width, coarse_step)
    _, _, refined = best_direction(u, v, 2.0 * coarse_step, fine_step)

    return refined


# Matched-filter pilot SNR (M * L * SNR_coherent / N) needed for the
# direction search to pick the true peak among ~1e3 candidates. Over 10
# trials per range, 32 and 64 still gave rare sidelobe locks (worst 46 and
# 24 dB loss at 50 m); 128 kept the worst case under 0.3 dB.
REQUIRED_PILOT_SNR = 128.0
PILOT_PATTERNS = 64


def adaptive_pilot_plan(
    prior_m: np.ndarray,
    prior_std_m: float,
    config: SimulationConfig,
) -> dict:
    """
    Choose the search width and pilot budget from the prior alone.

    The coherent SNR at the prior position is predicted from the known
    BS/RIS geometry and hardware model. If the prior's angular uncertainty
    is already well inside the beam, no pilots are sent.
    """

    from channel import link_metrics

    offset_m = np.asarray(prior_m, dtype=float) - np.asarray(config.ris_position_m, dtype=float)
    angular_std = prior_std_m / np.linalg.norm(offset_m)
    beam_width = 2.0 / max(config.ris_rows, config.ris_columns)

    if 3.0 * angular_std < beam_width / 8.0:
        return {"patterns": 0, "symbols_per_pattern": 0, "search_half_width": 0.0}

    predicted_snr = 10.0 ** (link_metrics(prior_m, config.ideal())["snr_db"] / 10.0)
    symbols = REQUIRED_PILOT_SNR * config.ris_elements / (PILOT_PATTERNS * max(predicted_snr, 1e-6))
    symbols = int(2 ** np.ceil(np.log2(max(symbols, 1.0))))

    return {
        "patterns": PILOT_PATTERNS,
        "symbols_per_pattern": symbols,
        "search_half_width": float(np.clip(3.0 * angular_std, 0.02, 0.5)),
    }


def estimate_from_prior(
    true_position_m: np.ndarray,
    prior_m: np.ndarray,
    prior_std_m: float,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Run the adaptive plan; return (position estimate, pilot symbols used)."""

    plan = adaptive_pilot_plan(prior_m, prior_std_m, config)
    if plan["patterns"] == 0:
        return np.asarray(prior_m, dtype=float), 0

    received, nominal = send_pilots(
        true_position_m,
        random_training(plan["patterns"], config, rng),
        config,
        plan["symbols_per_pattern"],
        rng,
    )
    estimate = refine_position(
        prior_m, received, nominal, config,
        search_half_width=plan["search_half_width"],
        coarse_step=min(0.02, plan["search_half_width"] / 4.0),
    )

    return estimate, plan["patterns"] * plan["symbols_per_pattern"]
