"""Tests for environment reward computation."""

from __future__ import annotations

import numpy as np

from milliampere_env.milliampere_env import MilliAmpereEnv
from milliampere_env.transport import MockTransport


def _make_env(configs_dir, config_name="legacy/v4_equivalent.yaml"):
    mock = MockTransport()
    env = MilliAmpereEnv(
        config_path=str(configs_dir / config_name),
        transport=mock,
    )
    return env, mock


class TestRewardBasics:
    def test_reward_at_target_is_positive(self, configs_dir):
        env, mock = _make_env(configs_dir)
        env.reset(seed=42)
        # Place vessel exactly at target
        target = env._target_pose.copy()
        mock.set_pose(*target)
        _, reward, _, _, _ = env.step(np.zeros(8))
        assert reward > 0

    def test_reward_far_from_target_is_lower(self, configs_dir):
        env, mock = _make_env(configs_dir)
        env.reset(seed=42)
        target = env._target_pose.copy()

        # Near target
        mock.set_pose(*target)
        _, reward_near, _, _, _ = env.step(np.zeros(8))

        env.reset(seed=42)
        # Far from target (5m offset)
        mock.set_pose(target[0] + 5.0, target[1], target[2])
        _, reward_far, _, _, _ = env.step(np.zeros(8))

        assert reward_near > reward_far

    def test_termination_penalty(self, configs_dir):
        env, mock = _make_env(configs_dir)
        env.reset(seed=42)
        # Place vessel beyond termination distance
        mock.set_pose(15.0, 0.0, 0.0)
        _, reward, terminated, _, info = env.step(np.zeros(8))
        assert terminated is True
        assert reward < -50  # termination penalty is -100


class TestAngleRatePenalty:
    def test_disabled_when_weight_zero(self, configs_dir):
        env, mock = _make_env(configs_dir, "legacy/v4_equivalent.yaml")
        env.reset(seed=42)
        mock.set_pose(0.0, 0.0, 0.0)
        env.step(np.zeros(8))
        assert env._R_angle_d == 0.0

    def test_enabled_when_weight_nonzero(self, configs_dir):
        env, mock = _make_env(configs_dir, "legacy/v9_equivalent.yaml")
        env.reset(seed=42)
        mock.set_pose(0.0, 0.0, 0.0)
        # First step with zero action to establish baseline angles
        env.step(np.zeros(8))
        # Force current angles to differ so next step sees a delta
        env._angles[:] = [90.0, -90.0, 0.0, 0.0]
        # Step — prev_angles will be [90, -90, 0, 0], new angles from action will differ
        action = np.array([-0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5])
        env.step(action)
        # Angle rate penalty should be negative (penalizing angle changes)
        assert env._R_angle_d < 0.0


class TestRewardComponents:
    def test_info_contains_reward_components(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        rc = info["reward_components"]
        assert "R_gauss" in rc
        assert "R_AS_gauss" in rc
        assert "R_vel" in rc
        assert "R_thrust" in rc
        assert "R_thrust_d" in rc
        assert "R_angle_d" in rc

    def test_gaussian_reward_non_negative(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["reward_components"]["R_gauss"] >= 0
        assert info["reward_components"]["R_AS_gauss"] >= 0

    def test_thrust_penalty_non_positive(self, configs_dir):
        env, _ = _make_env(configs_dir)
        env.reset(seed=42)
        # Use a non-zero action to generate thrust
        action = np.array([-0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5])
        _, _, _, _, info = env.step(action)
        assert info["reward_components"]["R_thrust"] <= 0
