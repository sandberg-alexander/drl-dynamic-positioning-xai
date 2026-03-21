"""Pydantic configuration models for training, evaluation, and deployment."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """Configuration for DRL training runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Reproducibility
    seed: int = Field(default=42)

    # Environment
    env_config: str = "configs/env/dp_positive_thrust.yaml"

    # PPO hyperparameters
    policy: str = "MlpPolicy"
    n_steps: int = Field(default=2048, gt=0)
    total_timesteps: int = Field(default=500_000, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=64, gt=0)
    n_epochs: int = Field(default=10, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1)
    device: str = "cpu"

    # Checkpointing
    save_interval: int = Field(default=2, ge=1)

    # W&B (optional)
    wandb_project: str | None = None
    wandb_offline: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class EvalConfig(BaseModel):
    """Configuration for model evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(default=42)
    env_config: str = "configs/env/legacy/v5_equivalent.yaml"
    n_eval_episodes: int = Field(default=10, gt=0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class DeployConfig(BaseModel):
    """Configuration for DRL deployment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_path: str = "/app/models/training_20250404_165037/models/best_model.zip"
    env_config: str = "configs/env/legacy/v10_equivalent.yaml"

    # Test parameters
    vep_length_dp: int = 200
    test_length_dp: int = 5
    test_length_north: int = 200
    test_length_spline: int = 200

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeployConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
