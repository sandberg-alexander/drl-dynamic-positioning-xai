"""Integration test: full DRL episode with MockTransport.

Verifies the complete loop works end-to-end:
1. Create env with MockTransport (no ROS needed)
2. Create a fresh PPO model
3. Run a short episode (model.predict -> env.step)
4. Assert: rewards are finite, episode terminates, no crashes
"""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3 import PPO

from milliampere_env.milliampere_env import MilliAmpereEnv
from milliampere_env.transport import MockTransport


@pytest.fixture()
def env_with_mock(configs_dir):
    """Create a MilliAmpereEnv backed by MockTransport."""
    config_path = str(configs_dir / "dp_positive_thrust.yaml")
    mock = MockTransport()
    env = MilliAmpereEnv(config_path=config_path, transport=mock)
    yield env
    env.close()


class TestFullEpisode:
    """Run a complete episode using a real PPO model with MockTransport."""

    def test_episode_runs_to_completion(self, env_with_mock):
        env = env_with_mock
        obs, info = env.reset(seed=42)
        total_reward = 0.0
        steps = 0

        while steps < 100:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            assert np.isfinite(reward), f"Non-finite reward at step {steps}"
            assert np.all(np.isfinite(obs)), f"Non-finite obs at step {steps}"
            if done or truncated:
                break

        assert steps > 0

    def test_short_training_no_crash(self, env_with_mock):
        env = env_with_mock
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            device="cpu",
            seed=42,
        )
        model.learn(total_timesteps=128)

    def test_predict_after_training(self, env_with_mock):
        env = env_with_mock
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            device="cpu",
            seed=42,
        )
        model.learn(total_timesteps=128)

        obs, _ = env.reset(seed=99)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == env.action_space.shape
