"""Combine synthetic and public UAV trajectories."""

from pathlib import Path

import pandas as pd

from config import SimulationConfig
from dataset_generator import (
    generate_designed_trajectories,
    generate_random_trajectories,
    write_trajectory_dataset,
)


def main() -> None:
    """Generate the hybrid UAV/RIS dataset."""

    config = SimulationConfig()

    designed = (
        generate_designed_trajectories(
            config
        )
    )

    random = (
        generate_random_trajectories(
            config=config,
            number_of_trajectories=91,
            seed=42,
            starting_id=10,
        )
    )

    synthetic_trajectories = (
        designed + random
    )

    output_path = (
        Path(__file__).parent
        / "results"
        / "hybrid_ris_dataset.csv"
    )

    write_trajectory_dataset(
        trajectories=synthetic_trajectories,
        output_path=output_path,
    )

    dataframe = pd.read_csv(
        output_path
    )

    dataframe.insert(
        1,
        "trajectory_source",
        "synthetic",
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        "Hybrid dataset currently contains "
        "the synthetic trajectory sources."
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Trajectories: "
        f"{dataframe['trajectory_id'].nunique()}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()