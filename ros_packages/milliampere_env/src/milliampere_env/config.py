"""Pydantic config schema for the MilliAmpere1 environment.

All YAML configs are validated at load time. Frozen models prevent
accidental mutation during training. Typos and invalid values fail
immediately with clear error messages.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TargetType(str, Enum):
    random = "random"
    waypoint = "waypoint"


class TargetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: TargetType = TargetType.random
    bounds: Tuple[float, float, float] = (5.0, 5.0, 180.0)


class ResetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_service: bool = False
    num_calls: int = Field(default=1, ge=1, le=3)
    sleep_between_s: float = Field(default=3.0, ge=0.0)


class RewardConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    w_gauss: float = Field(default=1.0, ge=0.0)
    w_AS_gauss: float = Field(default=0.4, ge=0.0)
    sigma_d: float = Field(default=1.0, gt=0.0)
    sigma_psi: float = Field(default=250.0, gt=0.0)
    sigma_AS_d: float = Field(default=25.0, gt=0.0)
    sigma_AS_psi: float = Field(default=3500.0, gt=0.0)
    velocity_weight: float = Field(default=0.1, ge=0.0)
    thrust_weight: float = Field(default=0.1, ge=0.0)
    thrust_rate_weight: float = Field(default=0.1, ge=0.0)
    angle_rate_weight: float = Field(default=0.0, ge=0.0)
    termination_penalty: float = Field(default=-100.0, le=0.0)


class ModeCheckingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    topic: str = "/supervisor/mode"
    required_mode: str = "drl"


class EnvConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action_space_type: Literal["positive_only"] = "positive_only"
    max_time_steps: int = Field(default=3000, gt=0)
    dt: float = Field(default=0.1, gt=0.0)
    target: TargetConfig = TargetConfig()
    reset: ResetConfig = ResetConfig()
    rewards: RewardConfig = RewardConfig()
    mode_checking: ModeCheckingConfig = ModeCheckingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnvConfig:
        """Load and validate config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
