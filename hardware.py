"""Hardware impairment models for the BS transmitter and the RIS.

Placement of each impairment follows where it physically occurs:

- PA nonlinearity (Saleh) is at the BS transmitter. A passive RIS has no
  power amplifier. Its effect on a modulated (OFDM-like, Gaussian) signal
  is modelled with the Bussgang decomposition: the PA output is a scaled
  copy of the input plus uncorrelated distortion that travels through the
  channel with the signal. This caps the SNDR no matter how strong the
  link is, which a fixed dB loss cannot capture.
- RIS phase quantization, phase noise and amplitude-phase coupling act on
  every reflecting element.
- S11 mismatch is a loss at the BS and UAV antenna ports.
"""

from functools import lru_cache

import numpy as np

from config import SimulationConfig

# Saleh (1981) TWT coefficients.
SALEH_ALPHA_A = 2.1587
SALEH_BETA_A = 1.1517
SALEH_ALPHA_PHI = 4.0033
SALEH_BETA_PHI = 9.1040

# Practical RIS phase-shift model (Abeywickrama et al., IEEE TCOM 2020).
# beta_min is per-config (SimulationConfig.ris_coupling_beta_min) because no
# measured 60 GHz value is available; run_physical_layer sweeps it.
COUPLING_PHI = 0.43 * np.pi
COUPLING_KAPPA = 1.6


def saleh_pa(
    signal: np.ndarray,
) -> np.ndarray:
    """Apply the Saleh AM/AM and AM/PM model to a complex baseband signal."""

    amplitude = np.abs(signal)
    amplitude_out = SALEH_ALPHA_A * amplitude / (1.0 + SALEH_BETA_A * amplitude**2)
    phase_shift = SALEH_ALPHA_PHI * amplitude**2 / (1.0 + SALEH_BETA_PHI * amplitude**2)

    return amplitude_out * np.exp(1j * (np.angle(signal) + phase_shift))


@lru_cache(maxsize=32)
def pa_bussgang_factors(
    input_backoff_db: float,
    samples: int = 200_000,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Return (useful_fraction, distortion_fraction) of the PA output power.

    The input is complex Gaussian with power set by the input back-off
    from the Saleh saturation point (1 / beta_a). Output power is
    normalised to 1, so the transmit power refers to the radiated total.
    """

    rng = np.random.default_rng(seed)

    input_power = 10.0 ** (-input_backoff_db / 10.0) / SALEH_BETA_A

    x = np.sqrt(input_power / 2.0) * (
        rng.standard_normal(samples) + 1j * rng.standard_normal(samples)
    )

    y = saleh_pa(x)

    bussgang_gain = np.mean(y * np.conj(x)) / np.mean(np.abs(x) ** 2)
    output_power = np.mean(np.abs(y) ** 2)
    useful_power = np.abs(bussgang_gain) ** 2 * np.mean(np.abs(x) ** 2)

    useful_fraction = float(useful_power / output_power)

    return useful_fraction, 1.0 - useful_fraction


def transmitter_factors(
    config: SimulationConfig,
) -> tuple[float, float]:
    """Return (useful, distortion) power fractions of the BS transmitter."""

    if not config.use_pa_nonlinearity:
        return 1.0, 0.0

    return pa_bussgang_factors(config.pa_input_backoff_db)


def s11_power_factor(
    config: SimulationConfig,
) -> float:
    """Return the mismatch power transfer 1 - |S11|^2 at one antenna port."""

    if not config.use_s11_mismatch:
        return 1.0

    reflection_magnitude = 10.0 ** (config.s11_db / 20.0)

    return 1.0 - reflection_magnitude**2


def quantize_phases(
    phases_rad: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Round phases to the nearest level of a uniform b-bit phase shifter."""

    step = 2.0 * np.pi / (2**bits)

    return np.round(np.mod(phases_rad, 2.0 * np.pi) / step) * step


def coupled_amplitude(
    phases_rad: np.ndarray,
    beta_min: float = 0.2,
) -> np.ndarray:
    """Return the phase-dependent reflection amplitude of a practical element."""

    return (
        (1.0 - beta_min)
        * ((np.sin(phases_rad - COUPLING_PHI) + 1.0) / 2.0) ** COUPLING_KAPPA
        + beta_min
    )


def hardware_aware_phases(
    cascaded_channel: np.ndarray,
    config: SimulationConfig,
    reference_phases: int = 64,
) -> np.ndarray:
    """
    Choose RIS phases knowing the hardware's quantization and coupling.

    For a common output phase psi, the best level for element n is the one
    maximising Re(c_n * Gamma(level) * e^{-j psi}). Sweeping psi and keeping
    the best combined amplitude is a standard near-optimal discrete-phase
    search, and it is fully vectorised.
    """

    if config.use_ris_phase_quantization:
        levels = np.arange(2**config.ris_phase_bits) * 2.0 * np.pi / 2**config.ris_phase_bits
    else:
        levels = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)

    gamma = np.exp(1j * levels)
    if config.use_ris_amplitude_phase_coupling:
        gamma = gamma * coupled_amplitude(levels, config.ris_coupling_beta_min)

    psi = np.linspace(0.0, 2.0 * np.pi, reference_phases, endpoint=False)

    # score[p, n, l] = Re(c_n * gamma_l * e^{-j psi_p})
    contributions = cascaded_channel[:, None] * gamma[None, :]
    scores = np.real(contributions[None, :, :] * np.exp(-1j * psi)[:, None, None])
    choice = np.argmax(scores, axis=2)

    combined = np.abs(np.take_along_axis(
        np.broadcast_to(contributions, (len(psi),) + contributions.shape),
        choice[:, :, None],
        axis=2,
    )[:, :, 0].sum(axis=1))

    return levels[choice[np.argmax(combined)]]


def ris_reflection_coefficients(
    commanded_phases_rad: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return the reflection coefficients the RIS actually realises.

    The controller commands continuous phases; the hardware quantizes
    them, adds phase noise, and couples amplitude to phase.
    """

    phases = np.asarray(commanded_phases_rad, dtype=float)

    if config.use_ris_phase_quantization:
        phases = quantize_phases(phases, config.ris_phase_bits)

    if config.use_ris_phase_noise:
        if rng is None:
            rng = np.random.default_rng(config.impairment_seed)
        phases = phases + rng.normal(
            0.0,
            np.radians(config.ris_phase_noise_std_deg),
            size=phases.shape,
        )

    amplitude = np.sqrt(config.ris_reflection_efficiency) * np.ones_like(phases)

    if config.use_ris_amplitude_phase_coupling:
        amplitude = amplitude * coupled_amplitude(phases, config.ris_coupling_beta_min)

    return amplitude * np.exp(1j * phases)
