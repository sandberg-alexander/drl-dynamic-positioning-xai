"""Smoke tests for SHAP processing pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import shap
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from milliampere_xai.model_wrappers import Obs2ActionWrapper


@pytest.fixture(scope="module")
def ppo_model():
    env = make_vec_env("CartPole-v1", n_envs=1)
    return PPO("MlpPolicy", env, n_steps=32, device="cpu")


class TestShapSmoke:
    def test_deep_explainer_produces_values(self, ppo_model):
        wrapper = Obs2ActionWrapper(ppo_model)
        background = torch.randn(10, 4)
        sample = torch.randn(1, 4)
        explainer = shap.DeepExplainer(wrapper, background)
        values = explainer.shap_values(sample)
        # SHAP returns array with shape (samples, obs_dim, action_dim)
        values_arr = np.asarray(values)
        assert values_arr.ndim >= 2
        # Last dim should be action count (2 for CartPole)
        assert values_arr.shape[-1] == 2
