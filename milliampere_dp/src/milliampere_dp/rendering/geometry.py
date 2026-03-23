"""Vessel geometry for rendering.

Pure functions that compute the milliAmpere1 hull polygon, forward
triangle, and moment marker line from physical dimensions.
"""

from __future__ import annotations

import numpy as np

from milliampere_dp.vessel import VESSEL_BEAM, VESSEL_LENGTH

# Rendering-specific geometry constants (metres)
VESSEL_CORNER: float = 0.5
VESSEL_TRIANGLE: float = 0.4
VESSEL_MOMENT_MARKER: float = 0.6


def vessel_hull_polygon(
    length: float = VESSEL_LENGTH,
    beam: float = VESSEL_BEAM,
    corner: float = VESSEL_CORNER,
) -> np.ndarray:
    """8-point hull polygon with rounded corners, centered at CoG.

    Returns shape (8, 2) in body-frame metres (x-forward, y-starboard).
    """
    hl = length / 2  # half-length
    hb = beam / 2  # half-beam
    c = corner
    return np.array(
        [
            [hl, -(hb - c)],
            [hl, (hb - c)],
            [hl - c, hb],
            [-(hl - c), hb],
            [-hl, (hb - c)],
            [-hl, -(hb - c)],
            [-(hl - c), -hb],
            [hl - c, -hb],
        ]
    )


def vessel_triangle_polygon(
    length: float = VESSEL_LENGTH,
    tri: float = VESSEL_TRIANGLE,
) -> np.ndarray:
    """Forward-pointing triangle at bow. Shape (3, 2)."""
    hl = length / 2
    return np.array(
        [
            [hl, 0.0],
            [hl - tri, tri / 2],
            [hl - tri, -tri / 2],
        ]
    )


def vessel_moment_marker_line(
    marker: float = VESSEL_MOMENT_MARKER,
) -> np.ndarray:
    """Moment marker line from CoG. Shape (2, 2)."""
    return np.array([[0.0, 0.0], [marker, 0.0]])
