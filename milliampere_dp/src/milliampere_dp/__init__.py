"""milliampere_dp -- shared domain library for milliAmpere1 dynamic positioning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("milliampere-dp")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from milliampere_dp import vessel as vessel  # type: ignore[attr-defined]
from milliampere_dp.log import get_logger
from milliampere_dp.rewards import (
    angle_rate_penalty,
    build_inv_sigma,
    gaussian_reward,
    thrust_penalty,
    thrust_rate_penalty,
    velocity_penalty,
)
from milliampere_dp.transforms import (
    quat2heading,
    rotate_body2ned,
    rotate_ned2body,
    ssa,
    ssa_alt,
    ssa_rad,
)

__all__ = [
    "__version__",
    "vessel",
    "get_logger",
    # rewards
    "build_inv_sigma",
    "gaussian_reward",
    "velocity_penalty",
    "thrust_penalty",
    "thrust_rate_penalty",
    "angle_rate_penalty",
    # transforms
    "rotate_ned2body",
    "rotate_body2ned",
    "ssa",
    "ssa_rad",
    "ssa_alt",
    "quat2heading",
]
