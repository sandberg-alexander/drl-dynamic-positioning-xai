"""Utilities for marking episode boundaries on time-series plots."""

from __future__ import annotations


def mark_episodes(
    ax,
    start: int,
    end: int,
    interval: int,
    color: str = "lightgray",
    alpha: float = 0.3,
) -> None:
    """Shade alternating episode spans on a time-series axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    start : int
        First timestep of the range.
    end : int
        Last timestep of the range (exclusive).
    interval : int
        Number of timesteps per episode.
    color : str
        Fill color for the shaded spans.
    alpha : float
        Opacity of the shaded spans.
    """
    shade = True
    for ep_start in range(start, end, interval):
        ep_end = min(ep_start + interval, end)
        if shade:
            ax.axvspan(ep_start, ep_end, color=color, alpha=alpha)
        shade = not shade
