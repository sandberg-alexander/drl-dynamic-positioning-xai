"""Tests for milliampere_dp.rewards."""

import numpy as np
import pytest

from milliampere_dp.rewards import (
    angle_rate_penalty,
    build_inv_sigma,
    gaussian_reward,
    thrust_penalty,
    thrust_rate_penalty,
    velocity_penalty,
)


class TestBuildInvSigma:
    def test_diagonal(self):
        inv = build_inv_sigma(1.0, 250.0)
        assert inv.shape == (2, 2)
        assert inv[0, 0] == pytest.approx(1.0)
        assert inv[1, 1] == pytest.approx(1.0 / 250.0)
        assert inv[0, 1] == pytest.approx(0.0)

    def test_invertibility(self):
        inv = build_inv_sigma(2.0, 100.0)
        sigma = np.diag([2.0, 100.0])
        np.testing.assert_allclose(inv @ sigma, np.eye(2), atol=1e-10)


class TestGaussianReward:
    @pytest.fixture()
    def inv_sigma(self):
        return build_inv_sigma(1.0, 250.0)

    def test_perfect_position_returns_weight(self, inv_sigma):
        reward = gaussian_reward(0.0, 0.0, inv_sigma, weight=1.0)
        assert reward == pytest.approx(1.0)

    def test_far_away_returns_near_zero(self, inv_sigma):
        reward = gaussian_reward(100.0, 0.0, inv_sigma, weight=1.0)
        assert reward < 0.01

    def test_weight_scales_output(self, inv_sigma):
        r1 = gaussian_reward(1.0, 10.0, inv_sigma, weight=1.0)
        r2 = gaussian_reward(1.0, 10.0, inv_sigma, weight=0.4)
        assert r2 == pytest.approx(0.4 * r1)

    def test_always_non_negative(self, inv_sigma):
        for d in [0, 1, 5, 10]:
            for psi in [0, 45, 90, 180]:
                assert gaussian_reward(d, psi, inv_sigma) >= 0.0

    def test_returns_float(self, inv_sigma):
        result = gaussian_reward(1.0, 10.0, inv_sigma)
        assert isinstance(result, float)


class TestVelocityPenalty:
    def test_zero_velocity_returns_zero(self):
        penalty = velocity_penalty(0.0, 0.0, 0.0)
        assert penalty == pytest.approx(0.0, abs=1e-10)

    def test_always_negative_for_nonzero_speed(self):
        assert velocity_penalty(0.5, 0.3, 0.1) < 0.0

    def test_returns_float(self):
        assert isinstance(velocity_penalty(0.5, 0.0, 0.0), float)


class TestThrustPenalty:
    def test_zero_thrust(self):
        penalty = thrust_penalty(np.zeros(4), 900.0)
        assert penalty == pytest.approx(0.0)

    def test_full_thrust(self):
        penalty = thrust_penalty(np.full(4, 900.0), 900.0)
        assert penalty == pytest.approx(-0.1)

    def test_half_thrust(self):
        penalty = thrust_penalty(np.full(4, 450.0), 900.0)
        assert penalty == pytest.approx(-0.05)

    def test_always_non_positive(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            thrusters = rng.uniform(0, 900, size=4)
            assert thrust_penalty(thrusters, 900.0) <= 0.0

    def test_returns_float(self):
        assert isinstance(thrust_penalty(np.zeros(4), 900.0), float)


class TestThrustRatePenalty:
    def test_no_change(self):
        t = np.array([100, 200, 300, 400], dtype=float)
        penalty = thrust_rate_penalty(t, t, 900.0)
        assert penalty == pytest.approx(0.0)

    def test_nonzero_change(self):
        t = np.array([100, 200, 300, 400], dtype=float)
        t_prev = np.zeros(4)
        penalty = thrust_rate_penalty(t, t_prev, 900.0)
        assert penalty < 0.0


class TestAngleRatePenalty:
    def test_no_change(self):
        a = np.array([135, -135, -45, 45], dtype=float)
        penalty = angle_rate_penalty(a, a)
        assert penalty == pytest.approx(0.0)

    def test_full_swing(self):
        a = np.array([90.0, 90.0, 90.0, 90.0])
        a_prev = np.array([0.0, 0.0, 0.0, 0.0])
        penalty = angle_rate_penalty(a, a_prev, max_angle_deg=90.0, weight=1.0)
        assert penalty == pytest.approx(-1.0)

    def test_weight_scales(self):
        a = np.array([90.0, 90.0, 90.0, 90.0])
        a_prev = np.zeros(4)
        p1 = angle_rate_penalty(a, a_prev, weight=1.0)
        p2 = angle_rate_penalty(a, a_prev, weight=0.5)
        assert p2 == pytest.approx(p1 / 2)
