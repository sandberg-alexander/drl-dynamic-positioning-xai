"""milliampere_drl -- DRL training, evaluation, and deployment for milliAmpere1 DP."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("milliampere-drl")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from milliampere_drl.config import DeployConfig, EvalConfig, TrainingConfig
from milliampere_drl.deployer import DeployMode

__all__ = [
    "__version__",
    "TrainingConfig",
    "EvalConfig",
    "DeployConfig",
    "DeployMode",
]
