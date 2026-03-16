"""Tests for Pydantic config schema."""

from __future__ import annotations


import pytest
from pydantic import ValidationError

from milliampere_env.config import EnvConfig, RewardConfig, TargetType


class TestEnvConfigDefaults:
    def test_default_config_loads(self):
        config = EnvConfig()
        assert config.max_time_steps == 3000
        assert config.dt == 0.1
        assert config.action_space_type == "positive_only"

    def test_frozen_rejects_mutation(self):
        config = EnvConfig()
        with pytest.raises(ValidationError):
            config.max_time_steps = 999

    def test_target_default_is_random(self):
        config = EnvConfig()
        assert config.target.type == TargetType.random

    def test_mode_checking_default_disabled(self):
        config = EnvConfig()
        assert config.mode_checking.enabled is False


class TestEnvConfigValidation:
    def test_negative_sigma_raises(self):
        with pytest.raises(ValidationError):
            EnvConfig(rewards=RewardConfig(sigma_d=-1.0))

    def test_zero_dt_raises(self):
        with pytest.raises(ValidationError):
            EnvConfig(dt=0.0)

    def test_negative_max_time_steps_raises(self):
        with pytest.raises(ValidationError):
            EnvConfig(max_time_steps=-1)

    def test_positive_termination_penalty_raises(self):
        with pytest.raises(ValidationError):
            EnvConfig(rewards=RewardConfig(termination_penalty=10.0))

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError):
            EnvConfig(unknown_field="oops")


class TestEnvConfigFromYaml:
    def test_v4_loads(self, configs_dir):
        config = EnvConfig.from_yaml(configs_dir / "legacy" / "v4_equivalent.yaml")
        assert config.rewards.angle_rate_weight == 0.0
        assert config.reset.use_service is False
        assert config.target.type == TargetType.random

    def test_v5_loads(self, configs_dir):
        config = EnvConfig.from_yaml(configs_dir / "legacy" / "v5_equivalent.yaml")
        assert config.reset.use_service is True
        assert config.reset.num_calls == 2
        assert config.reset.sleep_between_s == 3.0

    def test_v9_loads(self, configs_dir):
        config = EnvConfig.from_yaml(configs_dir / "legacy" / "v9_equivalent.yaml")
        assert config.rewards.angle_rate_weight == 0.1

    def test_v10_loads(self, configs_dir):
        config = EnvConfig.from_yaml(configs_dir / "legacy" / "v10_equivalent.yaml")
        assert config.target.type == TargetType.waypoint
        assert config.mode_checking.enabled is True
        assert config.mode_checking.required_mode == "drl"

    def test_dp_positive_thrust_loads(self, configs_dir):
        config = EnvConfig.from_yaml(configs_dir / "dp_positive_thrust.yaml")
        assert config.rewards.angle_rate_weight > 0

    def test_dp_waypoint_loads(self, configs_dir):
        config = EnvConfig.from_yaml(configs_dir / "dp_waypoint.yaml")
        assert config.target.type == TargetType.waypoint
