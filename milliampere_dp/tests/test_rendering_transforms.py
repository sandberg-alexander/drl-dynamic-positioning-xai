"""Tests for milliampere_dp.rendering.transforms."""

from __future__ import annotations

import numpy as np
import pytest

from milliampere_dp.rendering.transforms import (
    R2,
    T2,
    Viewport,
    degrees_to_pygame,
    meters_to_pixels,
    transform_body_to_ned,
    transform_shape,
)
from milliampere_dp.transforms import rotate_ned2body


class TestR2:
    def test_identity_at_zero(self):
        result = R2(0.0)
        np.testing.assert_allclose(result, np.eye(2), atol=1e-12)

    def test_90_degrees(self):
        result = R2(np.pi / 2)
        expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize(
        "psi", [0, np.pi / 6, np.pi / 4, np.pi / 2, -np.pi / 4, np.pi]
    )
    def test_orthogonal(self, psi):
        R = R2(psi)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-12)

    @pytest.mark.parametrize("deg", [0, 30, 45, 90, -45, 180])
    def test_matches_rotate_ned2body(self, deg):
        """R2(radians) should produce the same matrix as rotate_ned2body(degrees)."""
        rad = np.deg2rad(deg)
        np.testing.assert_allclose(R2(rad), rotate_ned2body(deg), atol=1e-12)

    def test_determinant_is_one(self):
        R = R2(1.23)
        assert np.linalg.det(R) == pytest.approx(1.0)


class TestT2:
    def test_shape(self):
        t = T2(3.0, 4.0)
        assert t.shape == (2, 1)

    def test_values(self):
        t = T2(3.0, 4.0)
        np.testing.assert_array_equal(t, np.array([[3.0], [4.0]]))


class TestViewport:
    def test_from_window_center(self):
        vp = Viewport.from_window(50.0, 1000.0, 800.0)
        assert vp.center == (500.0, 400.0)

    def test_from_window_scale(self):
        vp = Viewport.from_window(50.0, 1000.0, 800.0)
        assert vp.scale == 50.0

    def test_from_window_R_matrix(self):
        vp = Viewport.from_window(50.0, 1000.0, 800.0)
        expected_R = np.array([[0.0, 50.0], [-50.0, 0.0]])
        np.testing.assert_array_equal(vp.R, expected_R)

    def test_from_window_T_vector(self):
        vp = Viewport.from_window(50.0, 1000.0, 800.0)
        # T = [[center[1]], [center[0]]] = [[400], [500]]
        expected_T = np.array([[400.0], [500.0]])
        np.testing.assert_array_equal(vp.T, expected_T)

    def test_world_to_pixels_origin(self):
        """Origin (0, 0) should map to near-center pixels."""
        vp = Viewport.from_window(50.0, 1000.0, 800.0)
        result = vp.world_to_pixels(np.array([[0.0, 0.0]]))
        # (R @ [0,0].T + T).T = T.T = [[400, 500]]
        np.testing.assert_allclose(result, np.array([[400.0, 500.0]]))

    def test_world_to_pixels_known_point(self):
        vp = Viewport.from_window(50.0, 1000.0, 800.0)
        # Point (1, 0): R @ [1, 0].T = [0, -50].T
        # + T = [400, 450].T -> result = [400, 450]
        result = vp.world_to_pixels(np.array([[1.0, 0.0]]))
        np.testing.assert_allclose(result, np.array([[400.0, 450.0]]))

    def test_frozen(self):
        vp = Viewport.from_window(50.0, 1000.0, 800.0)
        with pytest.raises(AttributeError):
            vp.scale = 100.0  # type: ignore[misc]


class TestTransformShape:
    def test_identity_at_origin(self):
        shape = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = transform_shape(0.0, 0.0, 0.0, shape)
        np.testing.assert_allclose(result, shape, atol=1e-12)

    def test_translation_only(self):
        origin = np.array([[0.0, 0.0]])
        result = transform_shape(2.0, 3.0, 0.0, origin)
        np.testing.assert_allclose(result, np.array([[2.0, 3.0]]), atol=1e-12)

    def test_rotation_90(self):
        point = np.array([[1.0, 0.0]])
        result = transform_shape(0.0, 0.0, 90.0, point)
        # R2(pi/2).T @ [1, 0].T = [[0, -1], [1, 0]] @ [1, 0].T = [0, 1].T
        np.testing.assert_allclose(result, np.array([[0.0, 1.0]]), atol=1e-12)

    def test_accepts_degrees(self):
        """Callers pass degrees, conversion is internal."""
        shape = np.array([[1.0, 0.0]])
        result_45 = transform_shape(0.0, 0.0, 45.0, shape)
        assert result_45.shape == (1, 2)


class TestTransformBodyToNed:
    def test_identity(self):
        shape = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = transform_body_to_ned(0.0, 0.0, 0.0, shape)
        np.testing.assert_allclose(result, shape, atol=1e-12)

    def test_roundtrip(self):
        """transform_shape and transform_body_to_ned are not exact inverses
        because they use different math, but for zero translation they should
        produce related results."""
        shape = np.array([[1.0, 0.5], [-0.5, 1.0]])
        fwd = transform_shape(0.0, 0.0, 30.0, shape)
        back = transform_body_to_ned(0.0, 0.0, 30.0, fwd)
        np.testing.assert_allclose(back, shape, atol=1e-10)


class TestDegreesToPygame:
    def test_zero_degrees(self):
        assert degrees_to_pygame(0.0) == pytest.approx(-np.pi / 2)

    def test_90_degrees(self):
        assert degrees_to_pygame(90.0) == pytest.approx(0.0)

    def test_180_degrees(self):
        assert degrees_to_pygame(180.0) == pytest.approx(np.pi / 2)

    def test_270_degrees(self):
        assert degrees_to_pygame(270.0) == pytest.approx(np.pi)


class TestMetersToPixels:
    def test_scale_factor(self):
        assert meters_to_pixels(50.0, 2.0) == 100.0

    def test_zero(self):
        assert meters_to_pixels(50.0, 0.0) == 0.0
