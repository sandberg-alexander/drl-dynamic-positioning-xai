"""Tests for SB3 PPO model wrappers and thrust combination."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from milliampere_xai.model_wrappers import (
    Obs2ActionWrapper,
    Obs2ValueWrapper,
    combine_actuator_ref,
)


@pytest.fixture(scope="module")
def ppo_model():
    """Create a minimal PPO model for testing (CartPole, fast)."""
    env = make_vec_env("CartPole-v1", n_envs=1)
    model = PPO("MlpPolicy", env, n_steps=32, device="cpu")
    return model


class TestObs2ActionWrapper:
    def test_forward_returns_correct_shape(self, ppo_model):
        wrapper = Obs2ActionWrapper(ppo_model)
        obs = torch.randn(1, 4)  # CartPole obs dim
        result = wrapper(obs)
        assert result.shape == (1, 2)  # CartPole action dim

    def test_output_is_differentiable(self, ppo_model):
        wrapper = Obs2ActionWrapper(ppo_model)
        obs = torch.randn(1, 4, requires_grad=True)
        result = wrapper(obs)
        result.sum().backward()
        assert obs.grad is not None

    def test_batch_input(self, ppo_model):
        wrapper = Obs2ActionWrapper(ppo_model)
        obs = torch.randn(5, 4)
        result = wrapper(obs)
        assert result.shape == (5, 2)


class TestObs2ValueWrapper:
    def test_forward_returns_scalar_value(self, ppo_model):
        wrapper = Obs2ValueWrapper(ppo_model)
        obs = torch.randn(1, 4)
        result = wrapper(obs)
        assert result.shape == (1, 1)

    def test_batch_input(self, ppo_model):
        wrapper = Obs2ValueWrapper(ppo_model)
        obs = torch.randn(5, 4)
        result = wrapper(obs)
        assert result.shape == (5, 1)


class TestCombineActuatorRef:
    def test_zero_thrust_returns_zero(self):
        ref = [(0.0, 0.0)] * 4
        pos = np.array([[1.8, -0.8], [1.8, 0.8], [-1.8, 0.8], [-1.8, -0.8]])
        tot, angle, angular = combine_actuator_ref(ref, pos)
        assert tot == pytest.approx(0.0)
        assert angular == pytest.approx(0.0)

    def test_returns_three_values(self):
        ref = [(100.0, 45.0)] * 4
        pos = np.array([[1.8, -0.8], [1.8, 0.8], [-1.8, 0.8], [-1.8, -0.8]])
        result = combine_actuator_ref(ref, pos)
        assert len(result) == 3
