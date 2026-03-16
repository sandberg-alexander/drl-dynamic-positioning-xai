"""Tests for environment action/observation spaces."""

from __future__ import annotations

import numpy as np

from milliampere_env.milliampere_env import MilliAmpereEnv
from milliampere_env.transport import MockTransport


def _make_env(configs_dir, config_name="legacy/v4_equivalent.yaml"):
    return MilliAmpereEnv(
        config_path=str(configs_dir / config_name),
        transport=MockTransport(),
    )


class TestActionSpace:
    def test_shape_is_8(self, configs_dir):
        env = _make_env(configs_dir)
        assert env.action_space.shape == (8,)

    def test_positive_only_bounds(self, configs_dir):
        env = _make_env(configs_dir)
        expected_low = np.array([-1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0])
        expected_high = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(env.action_space.low, expected_low)
        np.testing.assert_array_equal(env.action_space.high, expected_high)

    def test_sample_is_valid(self, configs_dir):
        env = _make_env(configs_dir)
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)


class TestObservationSpace:
    def test_shape_is_14(self, configs_dir):
        env = _make_env(configs_dir)
        assert env.observation_space.shape == (14,)

    def test_state_dims_are_symmetric(self, configs_dir):
        env = _make_env(configs_dir)
        # First 6 dims (state) are [-1, 1]
        np.testing.assert_array_equal(env.observation_space.low[:6], -1.0)
        np.testing.assert_array_equal(env.observation_space.high[:6], 1.0)

    def test_action_dims_match_action_space(self, configs_dir):
        env = _make_env(configs_dir)
        # Last 8 dims match action space bounds
        np.testing.assert_array_equal(
            env.observation_space.low[6:], env.action_space.low
        )
        np.testing.assert_array_equal(
            env.observation_space.high[6:], env.action_space.high
        )

    def test_reset_obs_in_space(self, configs_dir):
        env = _make_env(configs_dir)
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs), f"obs out of bounds: {obs}"

    def test_step_obs_in_space(self, configs_dir):
        env = _make_env(configs_dir)
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"obs out of bounds: {obs}"
