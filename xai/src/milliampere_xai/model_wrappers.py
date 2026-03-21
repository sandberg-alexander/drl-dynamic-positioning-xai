"""SB3 PPO model wrappers for SHAP explanation and thrust combination."""

from __future__ import annotations

import math

import numpy as np
import torch


class Obs2ActionWrapper(torch.nn.Module):
    """Extract the policy (action) network from an SB3 PPO model for SHAP."""

    def __init__(self, model) -> None:
        super().__init__()
        self.mlp_extractor = model.policy.mlp_extractor.policy_net
        self.action_net = model.policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.mlp_extractor(obs)
        return self.action_net(x)


class Obs2ValueWrapper(torch.nn.Module):
    """Extract the value network from an SB3 PPO model for SHAP.

    Skips the FlattenExtractor (which contains nn.Flatten that
    SHAP's DeepExplainer doesn't recognise) and feeds observations
    directly into the MLP — valid for MlpPolicy with 1-D observations.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.mlp_extractor = model.policy.mlp_extractor.value_net
        self.value_net = model.policy.value_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_vf = self.mlp_extractor(obs)
        return self.value_net(latent_vf)


def combine_actuator_ref(
    actuator_ref: list[tuple[float, float]],
    actuator_pos: np.ndarray,
) -> tuple[float, float, float]:
    """Combine individual thruster references into total values.

    Combines into total thrust, angle, and angular thrust.

    Args:
        actuator_ref: List of (thrust, angle_deg) tuples per
            thruster.
        actuator_pos: Array of (x, y) positions per thruster in
            body frame.

    Returns:
        (total_thrust, total_angle_deg, total_angular_thrust)
    """
    thrust_x = 0.0
    thrust_y = 0.0
    tot_angular_thrust = 0.0

    for i, ((thrust, angle), (x, y)) in enumerate(zip(actuator_ref, actuator_pos)):
        rad = np.deg2rad(90 * (i + 1) - angle)
        angle_rad = angle * np.pi / 180

        thrust_x += thrust * math.cos(angle_rad)
        thrust_y += thrust * math.sin(angle_rad)

        if x * y < 0:
            ang_thrust = abs(x) * thrust * np.cos(rad) + abs(y) * thrust * np.sin(rad)
        else:
            ang_thrust = abs(x) * thrust * np.sin(rad) + abs(y) * thrust * np.cos(rad)
        tot_angular_thrust += ang_thrust

    tot_thrust = math.hypot(thrust_x, thrust_y)
    tot_angle = math.atan2(thrust_y, thrust_x) * 180 / np.pi

    return tot_thrust, tot_angle, tot_angular_thrust
