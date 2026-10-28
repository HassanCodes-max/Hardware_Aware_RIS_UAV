"""RIS geometry utilities."""

import numpy as np

from config import SimulationConfig


def ris_element_positions(
    config: SimulationConfig,
) -> np.ndarray:
    """
    Return the 3-D position of every RIS element.

    The RIS is an 8x8 planar array centered at ris_position_m.
    Elements are separated by half a wavelength.
    """

    rows = config.ris_rows
    columns = config.ris_columns
    spacing = config.ris_element_spacing_m

    center = np.array(
        config.ris_position_m,
        dtype=float,
    )

    row_offsets = (
        np.arange(rows)
        - (rows - 1) / 2.0
    ) * spacing

    column_offsets = (
        np.arange(columns)
        - (columns - 1) / 2.0
    ) * spacing

    positions = []

    for row_offset in row_offsets:
        for column_offset in column_offsets:
            position = center.copy()

            position[0] += column_offset
            position[1] += row_offset

            positions.append(position)

    return np.array(positions)


def ris_geometry_summary(
    config: SimulationConfig,
) -> dict:
    """Return basic information about the RIS geometry."""

    positions = ris_element_positions(config)

    return {
        "rows": config.ris_rows,
        "columns": config.ris_columns,
        "elements": config.ris_elements,
        "spacing_m": config.ris_element_spacing_m,
        "center_m": config.ris_position_m,
        "first_element_m": positions[0].tolist(),
        "last_element_m": positions[-1].tolist(),
    }