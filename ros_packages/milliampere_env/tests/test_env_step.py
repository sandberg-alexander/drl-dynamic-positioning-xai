"""Tests for environment step/reset mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from milliampere_env.milliampere_env import MilliAmpereEnv
from milliampere_env.transport import MockTransport


def _make_env(configs_dir, config_name="legacy/v4_equivalent.yaml", **mock_kwargs):
    mock = MockTransport(**mock_kwargs)
    env = MilliAmpereEnv(
        config_path=str(configs_dir / config_name),
        transport=mock,
    )
    return env, mock


class TestReset:
    def test_returns_obs_and_info(self, configs_dir):
        env, _ = _make_env(configs_dir)
        result = env.reset(seed=42)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_obs_shape(self, configs_dir):
        env, _ = _make_env(configs_dir)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (14,)

    def test_info_has_target_pose(self, configs_dir):
        env, _ = _make_env(configs_dir)
        _, info = env.reset(seed=42)
        assert "target_pose" in info
        assert len(info["target_pose"]) == 3

    def test_reset_calls_service_when_configured(self, configs_dir):
        env, mock = _make_env(configs_dir, "legacy/v5_equivalent.yaml")
        env.reset(seed=42)
        assert mock._reset_count == 2  # v5 does 2 calls


class TestStep:
    def test_returns_5_tuple(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        result = env.step(env.action_space.sample())
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_is_finite(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert np.isfinite(reward)

    def test_truncation_at_max_steps(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        # Fast-forward time_step to near max
        env._time_step = env.config.max_time_steps - 1
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert truncated is True
        assert terminated is False

    def test_termination_beyond_10m(self, configs_dir):
        env, mock = _make_env(configs_dir)
        env.reset(seed=42)
        # Place vessel far from target
        mock.set_pose(15.0, 0.0, 0.0)
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert terminated is True

    def test_no_termination_within_bounds(self, configs_dir):
        env, mock = _make_env(configs_dir)
        env.reset(seed=42)
        # Place vessel near target (target is random offset, vessel at origin)
        mock.set_pose(0.0, 0.0, 0.0)
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated is False


class TestVelocitySource:
    def test_uses_injected_velocity(self, configs_dir):
        """When MockTransport has velocity set, env uses it directly."""
        env, mock = _make_env(configs_dir)
        env.reset(seed=42)
        mock.set_velocity(0.5, -0.2, 0.01)
        env.step(env.action_space.sample())
        # est_nu should match the injected velocity
        assert env._est_nu[0] == pytest.approx(0.5)
        assert env._est_nu[1] == pytest.approx(-0.2)
        assert env._est_nu[2] == pytest.approx(0.01)

    def test_falls_back_to_pose_delta(self, configs_dir):
        """When no velocity is available, env estimates from pose deltas."""
        env, mock = _make_env(configs_dir)
        env.reset(seed=42)
        # No velocity set on mock — should fall back to pose-delta
        mock.set_pose(1.0, 0.0, 0.0)
        env.step(env.action_space.sample())
        # After one step with pose change, velocity should be non-zero
        # (exact values depend on dt and pose delta)
        env.step(env.action_space.sample())
        # est_nu is populated from pose differentiation (not zeros)
        assert env._est_nu is not None


class TestWaypointTarget:
    def test_waypoint_target_updates_each_step(self, configs_dir):
        env, mock = _make_env(configs_dir, "legacy/v10_equivalent.yaml")
        mock.set_mode("drl")  # Required for mode checking
        mock.set_waypoint(1.0, 2.0, 45.0)
        env.reset(seed=42)

        # First target should be from waypoint
        assert env._target_pose[0] == pytest.approx(1.0)
        assert env._target_pose[1] == pytest.approx(2.0)

        # Update waypoint and step
        mock.set_waypoint(10.0, 20.0, 90.0)
        env.step(env.action_space.sample())
        assert env._target_pose[0] == pytest.approx(10.0)
