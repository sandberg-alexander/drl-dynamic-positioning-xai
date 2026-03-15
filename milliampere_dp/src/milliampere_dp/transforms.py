"""Coordinate transforms for NED/body frame conversions and angle wrapping.

All functions are pure math extracted from the gymnasium environment files
where they were duplicated as instance methods across all 14 env versions.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotate_ned2body(psi_deg: float) -> np.ndarray:
    """2x2 rotation matrix from NED to body frame.

    Parameters
    ----------
    psi_deg : float
        Heading angle in degrees.

    Returns
    -------
    np.ndarray
        2x2 rotation matrix R such that v_body = R @ v_ned.
    """
    psi = np.deg2rad(psi_deg)
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s], [-s, c]])


def rotate_body2ned(psi_deg: float) -> np.ndarray:
    """2x2 rotation matrix from body to NED frame (transpose of ned2body)."""
    psi = np.deg2rad(psi_deg)
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def ssa(angle_deg: float) -> float:
    """Smallest signed angle in degrees. Maps to (-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def ssa_rad(angle_rad: float) -> float:
    """Smallest signed angle in radians. Maps to (-pi, pi]."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def ssa_alt(angle_rad: float) -> float:
    """SSA variant that maps -pi to +pi (used for actuator edge case)."""
    if angle_rad == -np.pi:
        return np.pi
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def quat2heading(quaternion: tuple | list | np.ndarray) -> float:
    """Extract heading (yaw) in degrees from an (x, y, z, w) quaternion.

    Uses scipy's Rotation with 'zxy' Euler convention, extracting
    the first angle (yaw around z-axis).
    """
    rotation = Rotation.from_quat(quaternion)
    return float(rotation.as_euler("zxy", degrees=True)[0])
