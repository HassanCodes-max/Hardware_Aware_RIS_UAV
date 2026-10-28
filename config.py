"""Simulation configuration for the hardware-aware RIS UAV model."""

from dataclasses import dataclass, replace

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_J_K = 1.380649e-23


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters shared by the trajectory, channel, and RIS models.

    Defaults describe a physically consistent link (v2). Use
    ``SimulationConfig.paper_v1()`` to reproduce the numbers reported in
    the first paper draft (no antenna gains, 8x8 RIS lying flat, isotropic
    elements, no impairments).
    """

    carrier_frequency_hz: float = 60e9
    bandwidth_hz: float = 100e6

    transmit_power_w: float = 0.1
    noise_figure_db: float = 7.0
    noise_temperature_k: float = 290.0

    # Transceiver antenna gains. The v1 model had none, which is why its
    # SNR was around -43 dB.
    bs_antenna_gain_dbi: float = 25.0
    uav_antenna_gain_dbi: float = 10.0

    ris_rows: int = 32
    ris_columns: int = 32
    ris_reflection_efficiency: float = 0.85
    ris_position_m: tuple = (0.0, 0.0, 20.0)
    # Outward normal of the RIS panel. (1, 0, 0) is a facade-mounted panel
    # facing +x. Only the half-space in front of the panel is served.
    ris_normal: tuple = (1.0, 0.0, 0.0)
    # Element gain 4*pi*A/lambda^2 with A = (lambda/2)^2 -> pi.
    use_element_gain: bool = True
    # cos^q(theta) element pattern, applied on both hops.
    use_obliquity: bool = True
    obliquity_exponent: float = 1.5

    bs_position_m: tuple = (25.0, -15.0, 10.0)

    uav_initial_position_m: tuple = (20.0, 0.0, 30.0)

    uav_speed_m_s: float = 10.0

    flight_duration_s: float = 30.0
    time_step_s: float = 0.01

    trajectory_direction: tuple = (1.0, 0.0, 0.0)

    atmospheric_pressure_hpa: float = 1013.25
    atmospheric_temperature_k: float = 288.15
    water_vapour_density_g_m3: float = 7.5

    # Hardware impairments (see hardware.py). Each one is switchable so the
    # ideal and hardware-aware channels can be compared on the same geometry.
    use_pa_nonlinearity: bool = True
    pa_input_backoff_db: float = 6.0
    use_ris_phase_quantization: bool = True
    ris_phase_bits: int = 2
    use_ris_phase_noise: bool = True
    ris_phase_noise_std_deg: float = 5.0
    use_ris_amplitude_phase_coupling: bool = True
    # Minimum reflection amplitude of the coupled element (1.0 = no coupling).
    ris_coupling_beta_min: float = 0.2
    use_s11_mismatch: bool = True
    s11_db: float = -10.0
    impairment_seed: int = 7

    @classmethod
    def paper_v1(cls, **overrides) -> "SimulationConfig":
        """Return the configuration used by the first paper draft."""

        legacy = cls(
            bs_antenna_gain_dbi=0.0,
            uav_antenna_gain_dbi=0.0,
            ris_rows=8,
            ris_columns=8,
            ris_normal=(0.0, 0.0, 1.0),
            use_element_gain=False,
            use_obliquity=False,
            bs_position_m=(0.0, 0.0, 0.0),
            uav_initial_position_m=(20.0, 0.0, 20.0),
            noise_figure_db=10.0 * np.log10(1e-12 / (BOLTZMANN_J_K * 290.0 * 100e6)),
            use_pa_nonlinearity=False,
            use_ris_phase_quantization=False,
            use_ris_phase_noise=False,
            use_ris_amplitude_phase_coupling=False,
            use_s11_mismatch=False,
        )

        return replace(legacy, **overrides)

    def ideal(self) -> "SimulationConfig":
        """Return the same geometry with every hardware impairment disabled."""

        return replace(
            self,
            use_pa_nonlinearity=False,
            use_ris_phase_quantization=False,
            use_ris_phase_noise=False,
            use_ris_amplitude_phase_coupling=False,
            use_s11_mismatch=False,
        )

    @property
    def noise_power_w(self) -> float:
        """Return receiver noise power kTB * NF in watts."""

        return (
            BOLTZMANN_J_K
            * self.noise_temperature_k
            * self.bandwidth_hz
            * 10.0 ** (self.noise_figure_db / 10.0)
        )

    @property
    def wavelength_m(self) -> float:
        """Return the carrier wavelength in metres."""

        return SPEED_OF_LIGHT_M_S / self.carrier_frequency_hz

    @property
    def ris_elements(self) -> int:
        """Return the total number of RIS elements."""

        return self.ris_rows * self.ris_columns

    @property
    def ris_element_spacing_m(self) -> float:
        """Return RIS element spacing using half-wavelength spacing."""

        return self.wavelength_m / 2.0

    @property
    def simulation_samples(self) -> int:
        """Return the expected number of trajectory samples."""

        return int(
            round(
                self.flight_duration_s
                / self.time_step_s
            )
        ) + 1
