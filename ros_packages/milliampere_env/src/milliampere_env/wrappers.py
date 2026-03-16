"""Gymnasium wrappers for the milliAmpere1 environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


class ClipReward(gym.RewardWrapper):
    """Clip reward to a bounded range."""

    def __init__(self, env: gym.Env, min_reward: float, max_reward: float) -> None:
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self.min_reward, self.max_reward))


class DiscreteActions(gym.ActionWrapper):
    """Map discrete action indices to continuous action vectors."""

    def __init__(self, env: gym.Env, disc_to_cont: list[np.ndarray]) -> None:
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act: int) -> np.ndarray:
        return self.disc_to_cont[act]
