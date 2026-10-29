"""Regression tests for the two-round beam search in beam_training_overhead.py.

Run from the repository root:  python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beam_training_overhead import (  # noqa: E402
    BEAM_MEASUREMENT_MS,
    BEAMS,
    REPORT_ROUND_TRIP_MS,
    graph_expanded_ranking,
    tie_graph,
    two_round_pick,
)


def sweeps_with_tie(samples: int, best: int, tie: int) -> np.ndarray:
    """Sweeps where `tie` is always 0.1 dB below `best` and the rest 10 dB below."""

    power = np.full((samples, BEAMS), 0.1)
    power[:, best] = 1.0
    power[:, tie] = 10 ** -0.01
    return power


def test_tie_graph_ranks_learned_ties_before_index_neighbours():
    graph = tie_graph(sweeps_with_tie(20, best=28, tie=38))
    assert graph[28, 0] == 38
    assert 28 not in graph[28, :BEAMS - 1]


def test_second_round_finds_a_distant_tie():
    # Training: 38 is often a near-tie of 28. Test: 38 is now the best, but
    # round 1 only looks around 28.
    graph = tie_graph(sweeps_with_tie(20, best=28, tie=38))
    power = sweeps_with_tie(1, best=38, tie=28)
    ranking = np.argsort(np.abs(np.arange(BEAMS) - 28), kind="stable")[None, :]

    one_round, _, _ = two_round_pick(ranking, 3, 0, np.inf, power, graph, power.max(axis=1))
    picked, measured, latency = two_round_pick(ranking, 3, 1, np.inf, power, graph, power.max(axis=1))

    assert one_round[0] == 28 and picked[0] == 38
    assert measured[0] == 4
    assert np.isclose(latency[0], 4 * BEAM_MEASUREMENT_MS + REPORT_ROUND_TRIP_MS)


def test_fallback_sweeps_everything_when_the_beam_is_lost():
    power = np.full((1, BEAMS), 1e-3)
    power[0, 60] = 1.0
    ranking = np.arange(BEAMS)[None, :]
    graph = tie_graph(sweeps_with_tie(5, best=0, tie=1))

    picked, measured, _ = two_round_pick(ranking, 2, 0, 3.0, power, graph, np.array([1.0]))
    assert picked[0] == 60 and measured[0] == BEAMS


def test_graph_expanded_ranking_adds_partners_after_the_first_beams():
    graph = tie_graph(sweeps_with_tie(20, best=28, tie=38))
    ranking = np.argsort(np.abs(np.arange(BEAMS) - 28), kind="stable")[None, :]
    expanded = graph_expanded_ranking(ranking, graph, first=2)

    assert list(expanded[0, :3]) == [28, 27, 38]
    assert sorted(expanded[0]) == list(range(BEAMS))
