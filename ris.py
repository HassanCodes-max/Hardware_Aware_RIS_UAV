"""RIS geometry utilities."""

import numpy as np

from config import SimulationConfig


def ris_normal(
    config: SimulationConfig,
) -> np.ndarray:
    """Return the unit outward normal of the RIS panel."""

    normal = np.asarray(
        config.ris_normal,
        dtype=float,
    )

    return normal / np.linalg.norm(normal)


def ris_panel_axes(
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the (column, row) unit vectors spanning the RIS panel.

    For a panel lying flat (normal along z) these are x and y, which
    matches the original element layout.
    """

    normal = ris_normal(config)

    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(normal[2]) > 0.9
        else np.array([0.0, 0.0, 1.0])
    )

    if abs(normal[2]) > 0.9:
        column_axis = reference - normal * (reference @ normal)
        column_axis /= np.linalg.norm(column_axis)
        row_axis = np.cross(normal, column_axis)
    else:
        row_axis = reference - normal * (reference @ normal)
        row_axis /= np.linalg.norm(row_axis)
        column_axis = np.cross(row_axis, normal)

    return column_axis, row_axis


def ris_element_positions(
    config: SimulationConfig,
) -> np.ndarray:
    """
    Return the 3-D position of every RIS element, shape (N, 3).

    The RIS is a rows x columns planar array centered at ris_position_m,
    lying in the plane perpendicular to ris_normal. Elements are
    separated by half a wavelength.
    """

    spacing = config.ris_element_spacing_m

    center = np.array(
        config.ris_position_m,
        dtype=float,
    )

    column_axis, row_axis = ris_panel_axes(config)

    row_offsets = (
        np.arange(config.ris_rows)
        - (config.ris_rows - 1) / 2.0
    ) * spacing

    column_offsets = (
        np.arange(config.ris_columns)
        - (config.ris_columns - 1) / 2.0
    ) * spacing

    row_grid, column_grid = np.meshgrid(
        row_offsets,
        column_offsets,
        indexing="ij",
    )

    return (
        center
        + column_grid.reshape(-1, 1) * column_axis
        + row_grid.reshape(-1, 1) * row_axis
    )


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