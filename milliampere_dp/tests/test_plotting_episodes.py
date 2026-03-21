"""Smoke tests for the mark_episodes plotting utility."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from milliampere_dp.plotting.episodes import mark_episodes


def test_mark_episodes_runs_without_error():
    """Verify the function runs on a basic axis without raising."""
    fig, ax = plt.subplots()
    ax.plot(range(1000), range(1000))
    mark_episodes(ax, 0, 1000, 200)
    plt.close(fig)


def test_mark_episodes_adds_patches():
    """Verify that patches are added to the axis."""
    fig, ax = plt.subplots()
    ax.plot(range(600), range(600))
    initial_patches = len(ax.patches)
    mark_episodes(ax, 0, 600, 200)
    assert len(ax.patches) > initial_patches
    plt.close(fig)


def test_mark_episodes_empty_range():
    """Verify no error when start == end."""
    fig, ax = plt.subplots()
    mark_episodes(ax, 0, 0, 200)
    plt.close(fig)
