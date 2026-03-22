"""Tests for read-only environment property accessors."""

from __future__ import annotations

import numpy as np

from milliampere_env.milliampere_env import MilliAmpereEnv
from milliampere_env.transport import MockTransport


def _make_env(configs_dir, config_name="legacy/v4_equivalent.yaml", **mock_kwargs):
    mock = MockTransport(**mock_kwargs)
    env = MilliAmpereEnv(
        config_path=str(configs_dir / config_name),
        transport=mock,
    )
    return env, mock


class TestTargetPose:
    def test_shape_and_dtype(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        tp = env.target_pose
        assert tp.shape == (3,)
        assert tp.dtype == np.float64

    def test_returns_copy(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        tp = env.target_pose
        tp[:] = 999.0
        assert not np.allclose(env.target_pose, 999.0)

    def test_matches_internal_state(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        np.testing.assert_array_equal(env.target_pose, env._target_pose)


class TestThrusters:
    def test_shape_and_dtype(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        t = env.thrusters
        assert t.shape == (4,)
        assert t.dtype == np.float64

    def test_returns_copy(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        t = env.thrusters
        t[:] = 999.0
        assert not np.allclose(env.thrusters, 999.0)

    def test_updates_after_step(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        # Non-zero action should produce valid thruster values
        action = env.action_space.sample()
        env.step(action)
        assert env.thrusters.shape == (4,)


class TestAngles:
    def test_shape_and_dtype(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        a = env.angles
        assert a.shape == (4,)
        assert a.dtype == np.float64

    def test_returns_copy(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        a = env.angles
        a[:] = 999.0
        assert not np.allclose(env.angles, 999.0)


class TestEpsilonNed:
    def test_shape_and_dtype(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        e = env.epsilon_ned
        assert e.shape == (3,)
        assert e.dtype == np.float64

    def test_returns_copy(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        e = env.epsilon_ned
        e[:] = 999.0
        assert not np.allclose(env.epsilon_ned, 999.0)

    def test_updates_after_step(self, configs_dir):
        env, mock = _make_env(configs_dir)
        env.reset(seed=42)
        e_before = env.epsilon_ned.copy()
        # Move vessel to change error
        mock.set_pose(3.0, 2.0, 10.0)
        env.step(env.action_space.sample())
        e_after = env.epsilon_ned
        assert not np.allclose(e_before, e_after)
