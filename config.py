"""Simulation configuration for the hardware-aware RIS UAV model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters shared by the trajectory, channel, and RIS models."""

    carrier_frequency_hz: float = 60e9
    bandwidth_hz: float = 100e6

    transmit_power_w: float = 0.1
    noise_power_w: float = 1e-12

    ris_rows: int = 8
    ris_columns: int = 8
    ris_reflection_efficiency: float = 0.85
    ris_position_m: tuple = (0.0, 0.0, 20.0)

    bs_position_m: tuple = (0.0, 0.0, 0.0)

    uav_initial_position_m: tuple = (20.0, 0.0, 20.0)

    uav_speed_m_s: float = 10.0

    flight_duration_s: float = 30.0
    time_step_s: float = 0.01

    trajectory_direction: tuple = (1.0, 0.0, 0.0)

    atmospheric_pressure_hpa: float = 1013.25
    atmospheric_temperature_k: float = 288.15
    water_vapour_density_g_m3: float = 7.5

    @property
    def wavelength_m(self) -> float:
        """Return the carrier wavelength in metres."""

        speed_of_light_m_s = 299_792_458.0

        return speed_of_light_m_s / self.carrier_frequency_hz

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