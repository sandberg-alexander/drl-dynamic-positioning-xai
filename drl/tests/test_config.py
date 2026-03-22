"""Tests for Pydantic training/eval/deploy configs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from milliampere_drl.config import DeployConfig, EvalConfig, TrainingConfig


class TestTrainingConfig:
    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.seed == 42
        assert cfg.n_steps == 2048
        assert cfg.total_timesteps == 500_000
        assert cfg.device == "cpu"
        assert cfg.policy == "MlpPolicy"

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "train.yaml"
        yaml_file.write_text("seed: 99\ntotal_timesteps: 100000\n")
        cfg = TrainingConfig.from_yaml(yaml_file)
        assert cfg.seed == 99
        assert cfg.total_timesteps == 100_000

    def test_frozen(self):
        cfg = TrainingConfig()
        with pytest.raises(ValidationError):
            cfg.seed = 99

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            TrainingConfig(unknown_field=True)

    def test_n_steps_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainingConfig(n_steps=0)

    def test_model_copy_override(self):
        cfg = TrainingConfig()
        cfg2 = cfg.model_copy(update={"device": "cpu"})
        assert cfg2.device == "cpu"
        assert cfg.device == "cpu"  # original unchanged


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.n_eval_episodes == 10

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "eval.yaml"
        yaml_file.write_text("seed: 7\nn_eval_episodes: 5\n")
        cfg = EvalConfig.from_yaml(yaml_file)
        assert cfg.seed == 7
        assert cfg.n_eval_episodes == 5


class TestDeployConfig:
    def test_defaults(self):
        cfg = DeployConfig()
        assert "v10_equivalent" in cfg.env_config

    def test_default_paths(self):
        cfg = DeployConfig()
        assert cfg.data_path_sim == "/app/runs/sim/"
        assert cfg.data_path_real == "/app/runs/real/"
        assert cfg.xai_action_sample_path == "/app/xai_samples/action/"
        assert cfg.xai_vf_sample_path == "/app/xai_samples/value_function/"

    def test_default_test_parameters(self):
        cfg = DeployConfig()
        assert len(cfg.test_pose) == 4
        assert len(cfg.sample_pose) == 4
        assert cfg.ds_north > 0
        assert cfg.ds_spline > 0
        assert cfg.vep_length_action == 200
        assert cfg.sample_length_action == 5
        assert cfg.sample_length_vf == 1000

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "deploy.yaml"
        yaml_file.write_text(
            "model_path: /app/models/test.zip\n"
            "env_config: configs/env/dp_waypoint.yaml\n"
        )
        cfg = DeployConfig.from_yaml(yaml_file)
        assert cfg.model_path == "/app/models/test.zip"

    def test_from_yaml_with_new_fields(self, tmp_path):
        yaml_file = tmp_path / "deploy.yaml"
        yaml_file.write_text(
            "model_path: /app/models/test.zip\n"
            "env_config: configs/env/dp_waypoint.yaml\n"
            "ds_north: 0.2\n"
            "sample_length_vf: 500\n"
        )
        cfg = DeployConfig.from_yaml(yaml_file)
        assert cfg.ds_north == 0.2
        assert cfg.sample_length_vf == 500
