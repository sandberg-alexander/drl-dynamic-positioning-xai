"""Tests for milliampere_dp.rendering.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from milliampere_dp.rendering.geometry import (
    VESSEL_CORNER,
    VESSEL_MOMENT_MARKER,
    VESSEL_TRIANGLE,
    vessel_hull_polygon,
    vessel_moment_marker_line,
    vessel_triangle_polygon,
)
from milliampere_dp.vessel import VESSEL_BEAM, VESSEL_LENGTH


class TestVesselHullPolygon:
    def test_shape(self):
        hull = vessel_hull_polygon()
        assert hull.shape == (8, 2)

    def test_symmetric_about_x_axis(self):
        """Y-coordinates should be symmetric (negate row order)."""
        hull = vessel_hull_polygon()
        y_coords = hull[:, 1]
        # Points 0-3 are the top half, 4-7 are the bottom half (mirrored)
        # Check that for each point, there's a corresponding negated y
        y_set = set(np.round(y_coords, 10))
        for y in y_set:
            if y != 0.0:
                assert -y in y_set

    def test_default_uses_vessel_constants(self):
        hull = vessel_hull_polygon()
        # Max x should be VESSEL_LENGTH/2
        assert np.max(hull[:, 0]) == pytest.approx(VESSEL_LENGTH / 2)
        # Max |y| should be VESSEL_BEAM/2
        assert np.max(np.abs(hull[:, 1])) == pytest.approx(VESSEL_BEAM / 2)

    def test_custom_dimensions(self):
        hull = vessel_hull_polygon(length=10.0, beam=4.0, corner=1.0)
        assert hull.shape == (8, 2)
        assert np.max(hull[:, 0]) == pytest.approx(5.0)
        assert np.max(np.abs(hull[:, 1])) == pytest.approx(2.0)

    def test_corner_radius_effect(self):
        """Corner points should be inset from the extremes."""
        hull = vessel_hull_polygon()
        hl = VESSEL_LENGTH / 2
        hb = VESSEL_BEAM / 2
        # Top-right corner is at (hl - VESSEL_CORNER, hb)
        assert any(np.allclose(p, [hl - VESSEL_CORNER, hb]) for p in hull)


class TestVesselTrianglePolygon:
    def test_shape(self):
        tri = vessel_triangle_polygon()
        assert tri.shape == (3, 2)

    def test_tip_at_bow(self):
        tri = vessel_triangle_polygon()
        assert tri[0, 0] == pytest.approx(VESSEL_LENGTH / 2)
        assert tri[0, 1] == pytest.approx(0.0)

    def test_custom_dimensions(self):
        tri = vessel_triangle_polygon(length=10.0, tri=0.8)
        assert tri[0, 0] == pytest.approx(5.0)
        assert tri[1, 0] == pytest.approx(4.2)


class TestVesselMomentMarkerLine:
    def test_shape(self):
        line = vessel_moment_marker_line()
        assert line.shape == (2, 2)

    def test_starts_at_origin(self):
        line = vessel_moment_marker_line()
        np.testing.assert_array_equal(line[0], [0.0, 0.0])

    def test_length(self):
        line = vessel_moment_marker_line()
        assert line[1, 0] == pytest.approx(VESSEL_MOMENT_MARKER)

    def test_custom_length(self):
        line = vessel_moment_marker_line(marker=1.5)
        assert line[1, 0] == pytest.approx(1.5)


class TestConstants:
    def test_vessel_corner(self):
        assert VESSEL_CORNER == 0.5

    def test_vessel_triangle(self):
        assert VESSEL_TRIANGLE == 0.4

    def test_vessel_moment_marker(self):
        assert VESSEL_MOMENT_MARKER == 0.6
