"""ITU-R P.676-13 atmospheric gaseous attenuation model."""

from pathlib import Path
import math


DATA_DIRECTORY = Path(__file__).resolve().parent / "p676_data"

PART1_FILE = DATA_DIRECTORY / "part1.txt"
PART2_FILE = DATA_DIRECTORY / "part2.txt"


# ITU-R P.676-13 Annex 1, Table 1
# Oxygen spectroscopic data:
# f0, a1, a2, a3, a4, a5, a6

OXYGEN_LINES = [
    (50.474214, 0.975, 9.651, 6.690, 0.0, 2.566, 6.850),
    (50.987745, 2.529, 8.653, 7.170, 0.0, 2.246, 6.800),
    (51.503360, 6.193, 7.709, 7.640, 0.0, 1.947, 6.729),
    (52.021429, 14.320, 6.819, 8.110, 0.0, 1.667, 6.640),
    (52.542418, 31.240, 5.983, 8.580, 0.0, 1.388, 6.526),
    (53.066934, 64.290, 5.201, 9.060, 0.0, 1.349, 6.206),
    (53.595775, 124.600, 4.474, 9.550, 0.0, 2.227, 5.085),
    (54.130025, 227.300, 3.800, 9.960, 0.0, 3.170, 3.750),
    (54.671180, 389.700, 3.182, 10.370, 0.0, 3.558, 2.654),
    (55.221384, 627.100, 2.618, 10.890, 0.0, 2.560, 2.952),
    (55.783815, 945.300, 2.109, 11.340, 0.0, -1.172, 6.135),
    (56.264774, 543.400, 0.014, 17.030, 0.0, 3.525, -0.978),
    (56.363399, 1331.800, 1.654, 11.890, 0.0, -2.378, 6.547),
    (56.968211, 1746.600, 1.255, 12.230, 0.0, -3.545, 6.451),
    (57.612486, 2120.100, 0.910, 12.620, 0.0, -5.416, 6.056),
    (58.323877, 2363.700, 0.621, 12.950, 0.0, -1.932, 0.436),
    (58.446588, 1442.100, 0.083, 14.910, 0.0, 6.768, -1.273),
    (59.164204, 2379.900, 0.387, 13.530, 0.0, -6.561, 2.309),
    (59.590983, 2090.700, 0.207, 14.080, 0.0, 6.957, -0.776),
    (60.306056, 2103.400, 0.207, 14.150, 0.0, -6.395, 0.699),
    (60.434778, 2438.000, 0.386, 13.390, 0.0, 6.342, -2.825),
    (61.150562, 2479.500, 0.621, 12.920, 0.0, 1.014, -0.584),
    (61.800158, 2275.900, 0.910, 12.630, 0.0, 5.014, -6.619),
    (62.411220, 1915.400, 1.255, 12.170, 0.0, 3.029, -6.759),
    (62.486253, 1503.000, 0.083, 15.130, 0.0, -4.499, 0.844),
    (62.997984, 1490.200, 1.654, 11.740, 0.0, 1.856, -6.675),
    (63.568526, 1078.000, 2.108, 11.340, 0.0, 0.658, -6.139),
    (64.127775, 728.700, 2.617, 10.880, 0.0, -3.036, -2.895),
    (64.678910, 461.300, 3.181, 10.380, 0.0, -3.968, -2.590),
    (65.224078, 274.000, 3.800, 9.960, 0.0, -3.528, -3.680),
    (65.764779, 153.000, 4.473, 9.550, 0.0, -2.548, -5.002),
    (66.302096, 80.400, 5.200, 9.060, 0.0, -1.660, -6.091),
    (66.836834, 39.800, 5.982, 8.580, 0.0, -1.680, -6.393),
    (67.369601, 18.560, 6.818, 8.110, 0.0, -1.956, -6.475),
    (67.900868, 8.172, 7.708, 7.640, 0.0, -2.216, -6.545),
    (68.431006, 3.397, 8.652, 7.170, 0.0, -2.492, -6.600),
    (68.960312, 1.334, 9.650, 6.690, 0.0, -2.773, -6.650),
    (118.750334, 940.300, 0.010, 16.640, 0.0, -0.439, 0.079),
    (368.498246, 67.400, 0.048, 16.400, 0.0, 0.0, 0.0),
    (424.763020, 637.700, 0.044, 16.400, 0.0, 0.0, 0.0),
    (487.249273, 237.400, 0.049, 16.000, 0.0, 0.0, 0.0),
    (715.392902, 98.100, 0.145, 16.000, 0.0, 0.0, 0.0),
    (773.839490, 572.300, 0.141, 16.200, 0.0, 0.0, 0.0),
    (834.145546, 183.100, 0.145, 14.700, 0.0, 0.0, 0.0),
]


# ITU-R P.676-13 Annex 1, Table 2
# Water-vapour spectroscopic data:
# f0, b1, b2, b3, b4, b5, b6

WATER_VAPOUR_LINES = [
    (22.235080, 0.1079, 2.144, 26.38, 0.76, 5.087, 1.00),
    (67.803960, 0.0011, 8.732, 28.58, 0.69, 4.930, 0.82),
    (119.995940, 0.0007, 8.353, 29.48, 0.70, 4.780, 0.79),
    (183.310087, 2.273, 0.668, 29.06, 0.77, 5.022, 0.85),
    (321.225630, 0.0470, 6.179, 24.04, 0.67, 4.398, 0.54),
    (325.152888, 1.514, 1.541, 28.23, 0.64, 4.893, 0.74),
    (336.227764, 0.0010, 9.825, 26.93, 0.69, 4.740, 0.61),
    (380.197353, 11.67, 1.048, 28.11, 0.54, 5.063, 0.89),
    (390.134508, 0.0045, 7.347, 21.52, 0.63, 4.810, 0.55),
    (437.346667, 0.0632, 5.048, 18.45, 0.60, 4.230, 0.48),
    (439.150807, 0.9098, 3.595, 20.07, 0.63, 4.483, 0.52),
    (443.018343, 0.1920, 5.048, 15.55, 0.60, 5.083, 0.50),
    (448.001085, 10.41, 1.405, 25.64, 0.66, 5.028, 0.67),
    (470.888999, 0.3254, 3.597, 21.34, 0.66, 4.506, 0.65),
    (474.689092, 1.260, 2.379, 23.20, 0.65, 4.804, 0.64),
    (488.490108, 0.2529, 2.852, 25.86, 0.69, 5.201, 0.72),
    (503.568532, 0.0372, 6.731, 16.12, 0.61, 3.980, 0.43),
    (504.482692, 0.0124, 6.731, 16.12, 0.61, 4.010, 0.45),
    (547.676440, 0.9785, 0.158, 26.00, 0.70, 4.500, 1.00),
    (552.020960, 0.1840, 0.158, 26.00, 0.70, 4.500, 1.00),
    (556.935985, 497.0, 0.159, 30.86, 0.69, 4.552, 1.00),
    (620.700807, 5.015, 2.391, 24.38, 0.71, 4.856, 0.68),
    (645.766085, 0.0067, 8.633, 18.00, 0.60, 4.000, 0.50),
    (658.005280, 0.2732, 7.816, 32.10, 0.69, 4.140, 1.00),
    (752.033113, 243.4, 0.396, 30.86, 0.68, 4.352, 0.84),
    (841.051732, 0.0134, 8.177, 15.90, 0.33, 5.760, 0.45),
    (859.965698, 0.1325, 8.055, 30.60, 0.68, 4.090, 0.84),
    (899.303175, 0.0547, 7.914, 29.85, 0.68, 4.530, 0.90),
    (902.611085, 0.0386, 8.429, 28.65, 0.70, 5.100, 0.95),
    (906.205957, 0.1836, 5.110, 24.08, 0.70, 4.700, 0.53),
    (916.171582, 8.400, 1.441, 26.73, 0.70, 5.150, 0.78),
    (923.112692, 0.0079, 10.293, 29.00, 0.70, 5.000, 0.80),
    (970.315022, 9.009, 1.919, 25.50, 0.64, 4.940, 0.67),
    (987.926764, 134.6, 0.257, 29.85, 0.68, 4.550, 0.90),
    (1780.000000, 17506.0, 0.952, 196.3, 2.00, 24.15, 5.00),
]


def _load_table(file_path):
    """Load an ITU coefficient file without NumPy."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"ITU-R P.676-13 data file not found: {file_path}"
        )

    data = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                data.append(
                    tuple(float(value) for value in parts)
                )
            except ValueError:
                continue

    return data


PART1_DATA = _load_table(PART1_FILE)
PART2_DATA = _load_table(PART2_FILE)


def _water_vapour_pressure_hpa(
    water_vapour_density_g_m3,
    temperature_k,
):
    """Calculate water-vapour partial pressure."""

    return (
        water_vapour_density_g_m3
        * temperature_k
        / 216.7
    )


def _oxygen_refractivity(
    frequency_ghz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """Calculate oxygen imaginary refractivity N''."""

    theta = 300.0 / temperature_k

    e = _water_vapour_pressure_hpa(
        water_vapour_density_g_m3,
        temperature_k,
    )

    p = pressure_hpa - e

    if p <= 0:
        raise ValueError(
            "Dry-air pressure must be greater than zero."
        )

    oxygen_refractivity = 0.0

    for (
        f0,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
    ) in OXYGEN_LINES:

        strength = (
            a1
            * 1e-7
            * p
            * theta ** 3
            * math.exp(
                a2 * (1.0 - theta)
            )
        )

        delta_f = (
            a3
            * 1e-4
            * (
                p
                * theta ** (0.8 - a4)
                + 1.1
                * e
                * theta
            )
        )

        delta_f = math.sqrt(
            delta_f ** 2
            + 2.25e-6
        )

        delta = (
            (a5 + a6 * theta)
            * 1e-4
            * (p + e)
            * theta ** 0.8
        )

        denominator_1 = (
            (f0 - frequency_ghz) ** 2
            + delta_f ** 2
        )

        denominator_2 = (
            (f0 + frequency_ghz) ** 2
            + delta_f ** 2
        )

        line_shape = (
            frequency_ghz
            / f0
            * (
                (
                    delta_f
                    - delta
                    * (
                        f0
                        - frequency_ghz
                    )
                )
                / denominator_1
                +
                (
                    delta_f
                    - delta
                    * (
                        f0
                        + frequency_ghz
                    )
                )
                / denominator_2
            )
        )

        oxygen_refractivity += (
            strength
            * line_shape
        )

    d = (
        5.6e-4
        * (p + e)
        * theta ** 0.8
    )

    dry_continuum = (
        frequency_ghz
        * p
        * theta ** 2
        * (
            (
                6.14e-5
                /
                (
                    d
                    * (
                        1.0
                        + (
                            frequency_ghz
                            / d
                        ) ** 2
                    )
                )
            )
            +
            (
                1.4e-12
                * p
                * theta ** 1.5
                /
                (
                    1.0
                    + 1.9e-5
                    * frequency_ghz ** 1.5
                )
            )
        )
    )

    return (
        oxygen_refractivity
        + dry_continuum
    )


def _water_vapour_refractivity(
    frequency_ghz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """Calculate water-vapour imaginary refractivity N''."""

    theta = 300.0 / temperature_k

    e = _water_vapour_pressure_hpa(
        water_vapour_density_g_m3,
        temperature_k,
    )

    p = pressure_hpa - e

    water_refractivity = 0.0

    for (
        f0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
    ) in WATER_VAPOUR_LINES:

        strength = (
            b1
            * 1e-1
            * e
            * theta ** 3.5
            * math.exp(
                b2 * (1.0 - theta)
            )
        )

        delta_f = (
            b3
            * 1e-4
            * (
                p
                * theta ** b4
                +
                b5
                * e
                * theta ** b6
            )
        )

        delta_f = math.sqrt(
            0.535 * delta_f ** 2
            +
            0.217 * delta_f
            +
            (
                2.1316e-12
                * f0 ** 2
                / theta
            )
        )

        denominator_1 = (
            (f0 - frequency_ghz) ** 2
            + delta_f ** 2
        )

        denominator_2 = (
            (f0 + frequency_ghz) ** 2
            + delta_f ** 2
        )

        line_shape = (
            frequency_ghz
            / f0
            * (
                delta_f / denominator_1
                +
                delta_f / denominator_2
            )
        )

        water_refractivity += (
            strength
            * line_shape
        )

    return water_refractivity


def specific_gaseous_attenuation_db_per_km(
    frequency_hz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """
    Calculate ITU-R P.676-13 Annex 1 specific gaseous attenuation.

    Returns total attenuation in dB/km.
    """

    if frequency_hz <= 0:
        raise ValueError(
            "Frequency must be greater than zero."
        )

    if pressure_hpa <= 0:
        raise ValueError(
            "Pressure must be greater than zero."
        )

    if temperature_k <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    if water_vapour_density_g_m3 < 0:
        raise ValueError(
            "Water-vapour density cannot be negative."
        )

    frequency_ghz = frequency_hz / 1e9

    if frequency_ghz > 1000.0:
        raise ValueError(
            "ITU-R P.676-13 Annex 1 is valid up to 1000 GHz."
        )

    oxygen_refractivity = _oxygen_refractivity(
        frequency_ghz,
        pressure_hpa,
        temperature_k,
        water_vapour_density_g_m3,
    )

    water_refractivity = _water_vapour_refractivity(
        frequency_ghz,
        pressure_hpa,
        temperature_k,
        water_vapour_density_g_m3,
    )

    oxygen_attenuation_db_per_km = (
        0.1820
        * frequency_ghz
        * oxygen_refractivity
    )

    water_attenuation_db_per_km = (
        0.1820
        * frequency_ghz
        * water_refractivity
    )

    return (
        oxygen_attenuation_db_per_km
        + water_attenuation_db_per_km
    )


def oxygen_specific_attenuation_db_per_km(
    frequency_hz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """Return oxygen/dry-air specific attenuation in dB/km."""

    frequency_ghz = frequency_hz / 1e9

    refractivity = _oxygen_refractivity(
        frequency_ghz,
        pressure_hpa,
        temperature_k,
        water_vapour_density_g_m3,
    )

    return (
        0.1820
        * frequency_ghz
        * refractivity
    )


def water_vapour_specific_attenuation_db_per_km(
    frequency_hz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """Return water-vapour specific attenuation in dB/km."""

    frequency_ghz = frequency_hz / 1e9

    refractivity = _water_vapour_refractivity(
        frequency_ghz,
        pressure_hpa,
        temperature_k,
        water_vapour_density_g_m3,
    )

    return (
        0.1820
        * frequency_ghz
        * refractivity
    )


def atmospheric_loss_db(
    distance_m,
    frequency_hz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """
    Calculate terrestrial-path atmospheric attenuation in dB.

    ITU-R P.676-13 Annex 1 equation (10):

        A = gamma * r0

    where gamma is in dB/km and r0 is path length in km.
    """

    if distance_m <= 0:
        raise ValueError(
            "Distance must be greater than zero."
        )

    specific_loss_db_per_km = (
        specific_gaseous_attenuation_db_per_km(
            frequency_hz,
            pressure_hpa,
            temperature_k,
            water_vapour_density_g_m3,
        )
    )

    distance_km = distance_m / 1000.0

    return (
        specific_loss_db_per_km
        * distance_km
    )


def atmospheric_loss_linear(
    distance_m,
    frequency_hz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """Return atmospheric attenuation as a linear power-loss ratio."""

    loss_db = atmospheric_loss_db(
        distance_m,
        frequency_hz,
        pressure_hpa,
        temperature_k,
        water_vapour_density_g_m3,
    )

    return 10.0 ** (
        loss_db / 10.0
    )


def oxygen_equivalent_height_km(
    frequency_hz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """Calculate P.676-13 Annex 2 oxygen equivalent height."""

    frequency_ghz = frequency_hz / 1e9

    if not PART1_DATA:
        raise ValueError(
            "Part 1 coefficient data is empty."
        )

    if (
        frequency_ghz < PART1_DATA[0][0]
        or frequency_ghz > PART1_DATA[-1][0]
    ):
        raise ValueError(
            "Frequency is outside the Part 1 range."
        )

    for index in range(
        len(PART1_DATA) - 1
    ):
        f1, a1, b1, c1, d1 = PART1_DATA[index]
        f2, a2, b2, c2, d2 = PART1_DATA[index + 1]

        if f1 <= frequency_ghz <= f2:

            if f1 == f2:
                return (
                    a1
                    + b1 * temperature_k
                    + c1 * pressure_hpa
                    + d1
                    * water_vapour_density_g_m3
                )

            ratio = (
                (frequency_ghz - f1)
                / (f2 - f1)
            )

            a = a1 + ratio * (a2 - a1)
            b = b1 + ratio * (b2 - b1)
            c = c1 + ratio * (c2 - c1)
            d = d1 + ratio * (d2 - d1)

            return (
                a
                + b * temperature_k
                + c * pressure_hpa
                + d
                * water_vapour_density_g_m3
            )

    raise ValueError(
        "Unable to interpolate Part 1."
    )


def water_vapour_coefficient(
    frequency_hz,
    pressure_hpa,
    temperature_k,
    water_vapour_density_g_m3,
):
    """Calculate P.676-13 Annex 2 K_V coefficient."""

    frequency_ghz = frequency_hz / 1e9

    if not PART2_DATA:
        raise ValueError(
            "Part 2 coefficient data is empty."
        )

    if (
        frequency_ghz < PART2_DATA[0][0]
        or frequency_ghz > PART2_DATA[-1][0]
    ):
        raise ValueError(
            "Frequency is outside the Part 2 range."
        )

    for index in range(
        len(PART2_DATA) - 1
    ):
        f1, a1, b1, c1, d1 = PART2_DATA[index]
        f2, a2, b2, c2, d2 = PART2_DATA[index + 1]

        if f1 <= frequency_ghz <= f2:

            if f1 == f2:
                return (
                    a1
                    + b1 * water_vapour_density_g_m3
                    + c1 * temperature_k
                    + d1 * pressure_hpa
                )

            ratio = (
                (frequency_ghz - f1)
                / (f2 - f1)
            )

            a = a1 + ratio * (a2 - a1)
            b = b1 + ratio * (b2 - b1)
            c = c1 + ratio * (c2 - c1)
            d = d1 + ratio * (d2 - d1)

            return (
                a
                + b * water_vapour_density_g_m3
                + c * temperature_k
                + d * pressure_hpa
            )

    raise ValueError(
        "Unable to interpolate Part 2."
    )


def p676_data_summary():
    """Return information about the loaded ITU coefficient files."""

    return {
        "part1_file": str(PART1_FILE),
        "part2_file": str(PART2_FILE),
        "part1_rows": len(PART1_DATA),
        "part2_rows": len(PART2_DATA),
        "part1_frequency_range_ghz": (
            PART1_DATA[0][0],
            PART1_DATA[-1][0],
        ),
        "part2_frequency_range_ghz": (
            PART2_DATA[0][0],
            PART2_DATA[-1][0],
        ),
        "oxygen_line_count": len(OXYGEN_LINES),
        "water_vapour_line_count": len(WATER_VAPOUR_LINES),
    }