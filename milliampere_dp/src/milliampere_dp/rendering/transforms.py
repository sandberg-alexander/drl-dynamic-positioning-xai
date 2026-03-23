"""2D affine transforms for vessel rendering.

Viewport encapsulates the NED-to-pixel affine (90-degree rotation +
scale + translation).  All transform helpers are pure functions.
Extracted from dashboard.py VesselRender methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Viewport:
    """NED-to-pixel affine transform.

    Parameters
    ----------
    scale : float
        Pixels per metre.
    center : tuple[float, float]
        Pixel centre of the window (x_px, y_px).
    R : np.ndarray
        2x2 rotation-scale matrix (90-degree CW rotation * scale).
    T : np.ndarray
        2x1 translation column vector (note: y,x order matching
        the original dashboard convention).
    """

    scale: float
    center: tuple[float, float]
    R: np.ndarray
    T: np.ndarray

    @classmethod
    def from_window(
        cls,
        scale: float,
        window_width: float,
        window_height: float,
    ) -> Viewport:
        """Create viewport from scale and window dimensions.

        Matches the dashboard VesselRender.__init__ convention::

            R = [[0, scale], [-scale, 0]]
            center = (width/2, height/2)
            T = [[center[1]], [center[0]]]   # note y,x order
        """
        center = (window_width / 2.0, window_height / 2.0)
        R = np.array([[0.0, scale], [-scale, 0.0]])
        T = np.array([[center[1]], [center[0]]])
        return cls(scale=scale, center=center, R=R, T=T)

    def world_to_pixels(self, coords: np.ndarray) -> np.ndarray:
        """Transform world (N, 2) coordinates to pixel coordinates.

        Equivalent to dashboard VesselRender._world_2_pixels::

            (R @ coords.T + T).T
        """
        return (self.R @ coords.T + self.T).T


# ---------------------------------------------------------------------------
# Pure transform functions
# ---------------------------------------------------------------------------


def R2(psi_rad: float) -> np.ndarray:
    """2x2 rotation matrix (NED convention).

    Parameters
    ----------
    psi_rad : float
        Heading angle in **radians**.

    Returns the same matrix as ``milliampere_dp.transforms.rotate_ned2body``
    but accepts radians directly (dashboard convention).
    """
    c, s = np.cos(psi_rad), np.sin(psi_rad)
    return np.array([[c, s], [-s, c]])


def T2(x: float, y: float) -> np.ndarray:
    """2x1 translation column vector.  Shape (2, 1)."""
    return np.array([[x], [y]])


def transform_shape(
    x: float,
    y: float,
    psi_deg: float,
    shape: np.ndarray,
) -> np.ndarray:
    """Rotate shape by -psi then translate by (x, y).

    Equivalent to dashboard VesselRender._transform_vessel.
    Accepts *degrees* (callers pass degrees, conversion is internal).

    Parameters
    ----------
    x, y : float
        Translation in body-frame metres.
    psi_deg : float
        Heading in **degrees**.
    shape : np.ndarray
        (N, 2) body-frame points.
    """
    psi_rad = np.deg2rad(psi_deg)
    return (R2(psi_rad).T @ shape.T + T2(x, y)).T


def transform_body_to_ned(
    x: float,
    y: float,
    psi_deg: float,
    shape: np.ndarray,
) -> np.ndarray:
    """Rotate shape by +psi after translating by (-x, -y).

    Equivalent to dashboard VesselRender._transform_body2ned.
    Accepts *degrees*.
    """
    psi_rad = np.deg2rad(psi_deg)
    return (R2(psi_rad) @ (shape.T + T2(-x, -y))).T


def degrees_to_pygame(angle_deg: float) -> float:
    """Convert degrees to pygame radians (90-degree offset).

    Equivalent to dashboard ``_degrees2pygame``::

        pi * angle / 180 - pi / 2
    """
    return np.pi * angle_deg / 180.0 - np.pi / 2.0


def meters_to_pixels(scale: float, value: float) -> float:
    """Convert metres to pixels.  Equivalent to ``_scalar2pygame``."""
    return scale * value
