"""Reward function components for DP station-keeping.

All functions are pure math -- no ROS, no gymnasium dependency.
Weights and covariance matrices are passed as arguments so that
a single set of functions serves all env versions.

Extracted from the gymnasium environment files (v1-v13) where
reward logic was duplicated with only weight values differing.
"""

from __future__ import annotations

import numpy as np


def build_inv_sigma(sigma_d: float, sigma_psi: float) -> np.ndarray:
    """Build the 2x2 inverse diagonal covariance matrix.

    Parameters
    ----------
    sigma_d : float
        Variance for distance error (must be > 0).
    sigma_psi : float
        Variance for heading error (must be > 0).

    Returns
    -------
    np.ndarray
        2x2 inverse covariance matrix.
    """
    return np.linalg.inv(np.diag([sigma_d, sigma_psi]))


def gaussian_reward(
    d: float,
    epsilon_psi: float,
    inv_sigma: np.ndarray,
    weight: float = 1.0,
) -> float:
    """Multivariate Gaussian position+heading reward.

    Parameters
    ----------
    d : float
        Euclidean distance error (metres).
    epsilon_psi : float
        Heading error (degrees). Absolute value is taken internally.
    inv_sigma : np.ndarray
        2x2 inverse covariance matrix.
    weight : float
        Scalar weight.

    Returns
    -------
    float
        Weighted Gaussian reward in [0, weight].
    """
    x = np.array([d, abs(epsilon_psi)])
    return float(weight * np.exp(-0.5 * x.T @ inv_sigma @ x))


def velocity_penalty(
    norm_u: float,
    norm_v: float,
    norm_r: float,
    weight: float = 0.1,
    base: float = 1000.0,
) -> float:
    """Exponential-decay velocity penalty.

    Penalises non-zero body-frame velocities with a base-1000
    exponential curve. Returns 0 at rest and approaches -weight at max speed.
    """
    speed = np.sqrt(norm_u**2 + norm_v**2 + norm_r**2)
    denom = base ** (-np.sqrt(3.0)) - 1.0
    return float(-weight * (base ** (-speed) - 1.0) / denom)


def thrust_penalty(
    thrusters: np.ndarray,
    max_rpm: float,
    weight: float = 0.1,
) -> float:
    """Penalty on absolute thrust magnitude (mean across thrusters)."""
    n = len(thrusters)
    return float(-weight / n * np.sum(np.abs(thrusters)) / max_rpm)


def thrust_rate_penalty(
    thrusters: np.ndarray,
    thrusters_prev: np.ndarray,
    max_rpm: float,
    weight: float = 0.1,
) -> float:
    """Penalty on thrust change rate between timesteps."""
    n = len(thrusters)
    return float(
        -weight / (2 * n) * np.sum(np.abs(thrusters - thrusters_prev)) / max_rpm
    )


def angle_rate_penalty(
    angles_deg: np.ndarray,
    angles_prev_deg: np.ndarray,
    max_angle_deg: float = 90.0,
    weight: float = 1.0,
) -> float:
    """Penalty on azimuth angle change rate between timesteps."""
    n = len(angles_deg)
    return float(
        -weight / n * np.sum(np.abs(angles_deg - angles_prev_deg)) / max_angle_deg
    )
