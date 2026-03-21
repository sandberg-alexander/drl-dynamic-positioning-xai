"""SB3 training callbacks for model saving, hyperparameter logging, and DP metrics."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class SaveModelCallback(BaseCallback):
    """Save models periodically after policy updates. Supports graceful stop."""

    def __init__(
        self, save_interval: int = 1, save_path: str | None = None, verbose: int = 1
    ) -> None:
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path
        self.policy_update_count = 0
        self._interrupted = False

    def request_stop(self) -> None:
        """Signal the callback to stop training after the current step."""
        self._interrupted = True

    def _on_rollout_end(self) -> None:
        self.policy_update_count += 1

        if self.policy_update_count % self.save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = (
                f"{self.save_path}/PPO_{timestamp}"
                f"_steps_{self.num_timesteps}"
                f"_update_{self.policy_update_count}"
            )
            self.model.save(model_path)

            if self.verbose > 0:
                print(f"Saved model at {model_path}")

                buf = self.model.ep_info_buffer
                if buf is not None and len(buf) > 0:
                    mean_reward = np.mean([ep_info["r"] for ep_info in buf])
                    print(f"Current mean reward: {mean_reward:.2f}")

    def _on_step(self) -> bool:
        if self._interrupted:
            return False
        return True


class HParamCallback(BaseCallback):
    """Log hyperparameters to TensorBoard HPARAMS tab on training start."""

    def _on_training_start(self) -> None:
        hparam_dict: dict[str, bool | str | float | None] = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": (
                float(self.model.learning_rate)
                if isinstance(self.model.learning_rate, float)
                else str(self.model.learning_rate)
            ),
        }
        if hasattr(self.model, "gamma"):
            hparam_dict["gamma"] = self.model.gamma  # type: ignore[attr-defined]
        if hasattr(self.model, "n_steps"):
            hparam_dict["n_steps"] = self.model.n_steps  # type: ignore[attr-defined]
        if hasattr(self.model, "seed") and self.model.seed is not None:
            hparam_dict["seed"] = self.model.seed

        metric_dict = {
            "rollout/ep_rew_mean": 0,
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class DPMetricsCallback(BaseCallback):
    """Log domain-specific DP metrics (position error, heading error, thrust)."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        if infos and isinstance(infos[0], dict):
            info = infos[0]
            if "position_error" in info:
                self.logger.record("dp/position_error_m", info["position_error"])
            if "heading_error" in info:
                self.logger.record("dp/heading_error_deg", info["heading_error"])
            if "total_thrust" in info:
                self.logger.record("dp/total_thrust_N", info["total_thrust"])
        return True
