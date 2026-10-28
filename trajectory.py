"""UAV trajectory generation utilities."""

from dataclasses import dataclass

import numpy as np

from config import SimulationConfig


@dataclass(frozen=True)
class Trajectory:
    """Sampled UAV positions, timestamps, and velocities."""

    time_s: np.ndarray
    positions_m: np.ndarray
    velocities_m_s: np.ndarray


def normalized_direction(
    direction: tuple | np.ndarray,
) -> np.ndarray:
    """Return a normalized 3-D direction vector."""

    direction = np.asarray(
        direction,
        dtype=float,
    )

    magnitude = np.linalg.norm(direction)

    if magnitude == 0:
        raise ValueError(
            "Trajectory direction cannot be zero."
        )

    return direction / magnitude


def straight_line_trajectory(
    config: SimulationConfig,
) -> Trajectory:
    """Generate a straight-line UAV trajectory."""

    time_s = np.arange(
        0.0,
        config.flight_duration_s
        + config.time_step_s / 2.0,
        config.time_step_s,
    )

    direction = normalized_direction(
        config.trajectory_direction
    )

    initial_position = np.array(
        config.uav_initial_position_m,
        dtype=float,
    )

    velocity = (
        config.uav_speed_m_s
        * direction
    )

    positions_m = (
        initial_position
        + time_s[:, np.newaxis]
        * velocity
    )

    velocities_m_s = np.tile(
        velocity,
        (len(time_s), 1),
    )

    return Trajectory(
        time_s=time_s,
        positions_m=positions_m,
        velocities_m_s=velocities_m_s,
    )


def straight_line_custom_trajectory(
    initial_position_m: tuple | np.ndarray,
    speed_m_s: float,
    direction: tuple | np.ndarray,
    duration_s: float,
    time_step_s: float,
) -> Trajectory:
    """Generate a custom straight-line trajectory."""

    if speed_m_s < 0:
        raise ValueError(
            "Speed cannot be negative."
        )

    time_s = np.arange(
        0.0,
        duration_s + time_step_s / 2.0,
        time_step_s,
    )

    initial_position = np.asarray(
        initial_position_m,
        dtype=float,
    )

    direction = normalized_direction(direction)

    velocity = speed_m_s * direction

    positions_m = (
        initial_position
        + time_s[:, np.newaxis]
        * velocity
    )

    velocities_m_s = np.tile(
        velocity,
        (len(time_s), 1),
    )

    return Trajectory(
        time_s=time_s,
        positions_m=positions_m,
        velocities_m_s=velocities_m_s,
    )


def circular_trajectory(
    center_m: tuple | np.ndarray,
    radius_m: float,
    altitude_m: float,
    angular_speed_rad_s: float,
    duration_s: float,
    time_step_s: float,
) -> Trajectory:
    """Generate a horizontal circular UAV trajectory."""

    if radius_m <= 0:
        raise ValueError(
            "Radius must be greater than zero."
        )

    time_s = np.arange(
        0.0,
        duration_s + time_step_s / 2.0,
        time_step_s,
    )

    center = np.asarray(
        center_m,
        dtype=float,
    )

    theta = angular_speed_rad_s * time_s

    positions_m = np.column_stack(
        (
            center[0] + radius_m * np.cos(theta),
            center[1] + radius_m * np.sin(theta),
            np.full_like(
                time_s,
                altitude_m,
            ),
        )
    )

    velocities_m_s = np.column_stack(
        (
            -radius_m
            * angular_speed_rad_s
            * np.sin(theta),
            radius_m
            * angular_speed_rad_s
            * np.cos(theta),
            np.zeros_like(time_s),
        )
    )

    return Trajectory(
        time_s=time_s,
        positions_m=positions_m,
        velocities_m_s=velocities_m_s,
    )


def helical_trajectory(
    center_m: tuple | np.ndarray,
    radius_m: float,
    initial_altitude_m: float,
    vertical_speed_m_s: float,
    angular_speed_rad_s: float,
    duration_s: float,
    time_step_s: float,
) -> Trajectory:
    """Generate a horizontal circular trajectory with changing altitude."""

    if radius_m <= 0:
        raise ValueError(
            "Radius must be greater than zero."
        )

    time_s = np.arange(
        0.0,
        duration_s + time_step_s / 2.0,
        time_step_s,
    )

    center = np.asarray(
        center_m,
        dtype=float,
    )

    theta = angular_speed_rad_s * time_s

    positions_m = np.column_stack(
        (
            center[0] + radius_m * np.cos(theta),
            center[1] + radius_m * np.sin(theta),
            initial_altitude_m
            + vertical_speed_m_s * time_s,
        )
    )

    velocities_m_s = np.column_stack(
        (
            -radius_m
            * angular_speed_rad_s
            * np.sin(theta),
            radius_m
            * angular_speed_rad_s
            * np.cos(theta),
            np.full_like(
                time_s,
                vertical_speed_m_s,
            ),
        )
    )

    return Trajectory(
        time_s=time_s,
        positions_m=positions_m,
        velocities_m_s=velocities_m_s,
    )


def waypoint_trajectory(
    waypoints_m: list | np.ndarray,
    speed_m_s: float,
    time_step_s: float,
) -> Trajectory:
    """
    Generate a trajectory through a sequence of 3-D waypoints.

    The UAV moves at approximately constant speed between
    consecutive waypoints.
    """

    if speed_m_s <= 0:
        raise ValueError(
            "Speed must be greater than zero."
        )

    waypoints = np.asarray(
        waypoints_m,
        dtype=float,
    )

    if len(waypoints) < 2:
        raise ValueError(
            "At least two waypoints are required."
        )

    positions = []
    velocities = []
    times = []

    current_time = 0.0

    for index in range(len(waypoints) - 1):

        start = waypoints[index]
        end = waypoints[index + 1]

        displacement = end - start
        distance = np.linalg.norm(displacement)

        if distance == 0:
            continue

        direction = displacement / distance

        segment_duration = distance / speed_m_s

        segment_times = np.arange(
            0.0,
            segment_duration,
            time_step_s,
        )

        for segment_time in segment_times:

            position = (
                start
                + direction
                * speed_m_s
                * segment_time
            )

            positions.append(position)
            velocities.append(
                direction * speed_m_s
            )
            times.append(
                current_time + segment_time
            )

        current_time += segment_duration

    positions.append(waypoints[-1])
    velocities.append(
        velocities[-1]
    )
    times.append(current_time)

    return Trajectory(
        time_s=np.asarray(times),
        positions_m=np.asarray(positions),
        velocities_m_s=np.asarray(velocities),
    )


def random_trajectory(
    rng: np.random.Generator,
    duration_s: float,
    time_step_s: float,
) -> Trajectory:
    """
    Generate one randomized but physically reasonable UAV trajectory.

    The trajectory type, initial position, speed, direction,
    and altitude are randomized.
    """

    trajectory_type = rng.choice(
        [
            "straight",
            "diagonal",
            "vertical",
            "circular",
            "helical",
            "waypoint",
        ]
    )

    initial_position = np.array(
        [
            rng.uniform(-50.0, 50.0),
            rng.uniform(-50.0, 50.0),
            rng.uniform(15.0, 60.0),
        ]
    )

    speed = rng.uniform(
        5.0,
        20.0,
    )

    if trajectory_type == "straight":

        direction = rng.normal(
            size=3
        )

        direction[2] *= 0.35

        return straight_line_custom_trajectory(
            initial_position_m=initial_position,
            speed_m_s=speed,
            direction=direction,
            duration_s=duration_s,
            time_step_s=time_step_s,
        )

    if trajectory_type == "diagonal":

        direction = np.array(
            [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-0.3, 0.3),
            ]
        )

        return straight_line_custom_trajectory(
            initial_position_m=initial_position,
            speed_m_s=speed,
            direction=direction,
            duration_s=duration_s,
            time_step_s=time_step_s,
        )

    if trajectory_type == "vertical":

        direction = np.array(
            [
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.choice([-1.0, 1.0]),
            ]
        )

        return straight_line_custom_trajectory(
            initial_position_m=initial_position,
            speed_m_s=speed,
            direction=direction,
            duration_s=duration_s,
            time_step_s=time_step_s,
        )

    if trajectory_type == "circular":

        radius = rng.uniform(
            20.0,
            80.0,
        )

        angular_speed = (
            speed / radius
        )

        return circular_trajectory(
            center_m=initial_position,
            radius_m=radius,
            altitude_m=initial_position[2],
            angular_speed_rad_s=angular_speed,
            duration_s=duration_s,
            time_step_s=time_step_s,
        )

    if trajectory_type == "helical":

        radius = rng.uniform(
            20.0,
            60.0,
        )

        angular_speed = (
            speed / radius
        )

        vertical_speed = rng.uniform(
            -2.0,
            2.0,
        )

        return helical_trajectory(
            center_m=initial_position,
            radius_m=radius,
            initial_altitude_m=initial_position[2],
            vertical_speed_m_s=vertical_speed,
            angular_speed_rad_s=angular_speed,
            duration_s=duration_s,
            time_step_s=time_step_s,
        )

    waypoint_count = rng.integers(
        3,
        7,
    )

    waypoints = [
        initial_position
    ]

    current = initial_position.copy()

    for _ in range(
        waypoint_count - 1
    ):

        current = current + np.array(
            [
                rng.uniform(-80.0, 80.0),
                rng.uniform(-80.0, 80.0),
                rng.uniform(-20.0, 20.0),
            ]
        )

        current[2] = np.clip(
            current[2],
            10.0,
            80.0,
        )

        waypoints.append(
            current.copy()
        )

    return waypoint_trajectory(
        waypoints_m=waypoints,
        speed_m_s=speed,
        time_step_s=time_step_s,
    )