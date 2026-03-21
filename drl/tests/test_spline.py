"""Tests for spline trajectory utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from milliampere_drl.spline import build_spline, heading_from_derivative


class TestBuildSpline:
    def test_returns_two_splines(self):
        sx, sy = build_spline()
        # Both should be callable at s=0 and s=1
        assert float(sx(0.0)) is not None
        assert float(sy(1.0)) is not None

    def test_start_at_origin(self):
        sx, sy = build_spline()
        assert float(sx(0.0)) == pytest.approx(0.0)
        assert float(sy(0.0)) == pytest.approx(0.0)

    def test_end_at_last_control_point(self):
        sx, sy = build_spline()
        assert float(sx(1.0)) == pytest.approx(20.0)
        assert float(sy(1.0)) == pytest.approx(0.0)

    def test_custom_control_points(self):
        pts = np.array([[0, 0], [5, 5], [10, 0]])
        sx, sy = build_spline(pts)
        assert float(sx(0.0)) == pytest.approx(0.0)
        assert float(sy(0.0)) == pytest.approx(0.0)
        assert float(sx(1.0)) == pytest.approx(10.0)
        assert float(sy(1.0)) == pytest.approx(0.0)

    def test_midpoint_is_smooth(self):
        sx, sy = build_spline()
        # Derivatives should exist and be finite at midpoint
        dx = float(sx(0.5, 1))
        dy = float(sy(0.5, 1))
        assert np.isfinite(dx)
        assert np.isfinite(dy)


class TestHeadingFromDerivative:
    def test_east_direction(self):
        # Moving east: dx>0, dy=0 -> heading = 0
        assert heading_from_derivative(1.0, 0.0) == pytest.approx(0.0)

    def test_north_direction(self):
        # Moving north: dx=0, dy>0 -> heading = pi/2
        assert heading_from_derivative(0.0, 1.0) == pytest.approx(math.pi / 2)

    def test_west_direction(self):
        result = heading_from_derivative(-1.0, 0.0)
        assert abs(result) == pytest.approx(math.pi)

    def test_south_direction(self):
        assert heading_from_derivative(0.0, -1.0) == pytest.approx(-math.pi / 2)
