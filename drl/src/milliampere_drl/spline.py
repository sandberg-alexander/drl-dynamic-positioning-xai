"""Trajectory spline utilities for path-following tests."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline

_DEFAULT_CONTROL_PTS = np.array(
    [
        [0.0, 0.0],  # p0
        [5.0, 2.0],  # p1
        [10.0, 0.0],  # p2
        [15.0, -3.0],  # p3
        [20.0, 0.0],  # p4
    ]
)


def build_spline(
    control_pts: np.ndarray | None = None,
) -> tuple[CubicSpline, CubicSpline]:
    """Return open cubic splines (x(s), y(s)) parameterised on s in [0, 1]."""
    if control_pts is None:
        control_pts = _DEFAULT_CONTROL_PTS
    t = np.linspace(0.0, 1.0, len(control_pts))
    spline_x = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    spline_y = CubicSpline(t, control_pts[:, 1], bc_type="natural")
    return spline_x, spline_y


def heading_from_derivative(dx: float, dy: float) -> float:
    """Return yaw [rad] from the path derivative (dx, dy)."""
    return math.atan2(dy, dx)
