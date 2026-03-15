"""milliAmpere1 vessel constants and thruster configuration.

All constants are extracted from the gymnasium environment files (v1-v13)
where they were duplicated identically across all versions.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical limits
# ---------------------------------------------------------------------------
MAX_THRUSTER_RPM: int = 900  # RPM
MAX_AZIMUTH_ANGLE: int = 90  # degrees
MAX_DISTANCE: float = 10.0  # metres (termination threshold)
MAX_HEADING_ANGLE: float = 180.0  # degrees
MAX_LINEAR_SPEED: float = 3.24  # m/s
MAX_ANGULAR_SPEED: float = 112.6  # degrees/s

# ---------------------------------------------------------------------------
# NED offset (Trondheim harbour, NTNU test site)
# ---------------------------------------------------------------------------
NED_NORTH_OFFSET: float = 334.61  # metres from NED origin
NED_EAST_OFFSET: float = 990.24  # metres from NED origin

# ---------------------------------------------------------------------------
# Vessel geometry
# ---------------------------------------------------------------------------
VESSEL_LENGTH: float = 5.06  # metres
VESSEL_BEAM: float = 2.86  # metres
THRUSTER_ARM_X: float = 1.8  # metres from CoG along x-body
THRUSTER_ARM_Y: float = 0.8  # metres from CoG along y-body

# ---------------------------------------------------------------------------
# Target bounds (max deviation from operating point in NED + heading)
# ---------------------------------------------------------------------------
TARGET_BOUNDS: np.ndarray = np.array([5.0, 5.0, 180.0])

# ---------------------------------------------------------------------------
# Thruster angle constraints (radians)
# ---------------------------------------------------------------------------
ACTUATOR_CONSTRAINTS: list[tuple[float, float]] = [
    (np.pi / 2, np.pi),  # Thruster 1: [90deg, 180deg]
    (-np.pi, -np.pi / 2),  # Thruster 2: [-180deg, -90deg]
    (-np.pi / 2, 0.0),  # Thruster 3: [-90deg, 0deg]
    (0.0, np.pi / 2),  # Thruster 4: [0deg, 90deg]
]

OPPOSITE_QUADRANTS: list[tuple[float, float]] = [
    (-np.pi / 2, 0.0),  # Opposite of Thruster 1
    (0.0, np.pi / 2),  # Opposite of Thruster 2
    (np.pi / 2, np.pi),  # Opposite of Thruster 3
    (-np.pi, -np.pi / 2),  # Opposite of Thruster 4
]

# ---------------------------------------------------------------------------
# Action space bounds (8-dimensional: 4 thrusters x 2 components)
# ---------------------------------------------------------------------------
LOW_ACTION: np.ndarray = np.array([-1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0])
HIGH_ACTION: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])

# ---------------------------------------------------------------------------
# Default initial thruster angles (degrees)
# ---------------------------------------------------------------------------
DEFAULT_INITIAL_ANGLES: np.ndarray = np.array([135.0, -135.0, -45.0, 45.0])
