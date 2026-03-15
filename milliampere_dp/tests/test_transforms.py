"""Tests for milliampere_dp.transforms."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from milliampere_dp.transforms import (
    quat2heading,
    rotate_body2ned,
    rotate_ned2body,
    ssa,
    ssa_alt,
    ssa_rad,
)


class TestRotateNed2Body:
    def test_identity_at_zero_heading(self):
        R = rotate_ned2body(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-15)

    def test_90_degrees(self):
        R = rotate_ned2body(90.0)
        expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_180_degrees(self):
        R = rotate_ned2body(180.0)
        expected = np.array([[-1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    @pytest.mark.parametrize("psi", [0, 30, 45, 90, -30, 135, 180])
    def test_roundtrip_is_identity(self, psi):
        R_n2b = rotate_ned2body(psi)
        R_b2n = rotate_body2ned(psi)
        np.testing.assert_allclose(R_n2b @ R_b2n, np.eye(2), atol=1e-10)

    def test_orthogonal(self):
        R = rotate_ned2body(37.0)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)


class TestSSA:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (0, 0),
            (360, 0),
            (270, -90),
            (-270, 90),
            (540, -180),  # modulo maps to -180, not +180
            (90, 90),
            (-90, -90),
        ],
    )
    def test_known_values(self, angle, expected):
        assert ssa(angle) == pytest.approx(expected, abs=1e-10)


class TestSSARad:
    def test_zero(self):
        assert ssa_rad(0.0) == pytest.approx(0.0)

    def test_two_pi_wraps_to_zero(self):
        assert ssa_rad(2 * np.pi) == pytest.approx(0.0, abs=1e-10)

    def test_three_pi_wraps_to_minus_pi(self):
        # modulo maps to -pi, not +pi (boundary behaviour)
        assert ssa_rad(3 * np.pi) == pytest.approx(-np.pi, abs=1e-10)

    def test_negative_three_pi(self):
        assert ssa_rad(-3 * np.pi) == pytest.approx(-np.pi, abs=1e-10)


class TestSSAAlt:
    def test_minus_pi_maps_to_plus_pi(self):
        assert ssa_alt(-np.pi) == pytest.approx(np.pi)

    def test_zero(self):
        assert ssa_alt(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_regular_values_match_ssa_rad(self):
        for angle in [0.5, 1.0, -1.0, 2.0]:
            assert ssa_alt(angle) == pytest.approx(ssa_rad(angle), abs=1e-10)


class TestQuat2Heading:
    def test_identity_quaternion(self):
        heading = quat2heading([0, 0, 0, 1])
        assert heading == pytest.approx(0.0, abs=1e-10)

    def test_90_degree_yaw(self):
        r = Rotation.from_euler("z", 90, degrees=True)
        q = r.as_quat()  # (x, y, z, w)
        heading = quat2heading(q)
        assert heading == pytest.approx(90.0, abs=1e-5)

    def test_minus_45_degree_yaw(self):
        r = Rotation.from_euler("z", -45, degrees=True)
        q = r.as_quat()
        heading = quat2heading(q)
        assert heading == pytest.approx(-45.0, abs=1e-5)

    def test_returns_float(self):
        result = quat2heading([0, 0, 0, 1])
        assert isinstance(result, float)
