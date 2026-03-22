"""milliampere_xai -- SHAP-based XAI explainer and dashboard for milliAmpere1 DRL-DP."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("milliampere-xai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from milliampere_xai.model_wrappers import (
    Obs2ActionWrapper,
    Obs2ValueWrapper,
    combine_actuator_ref,
)

__all__ = [
    "__version__",
    "Obs2ActionWrapper",
    "Obs2ValueWrapper",
    "combine_actuator_ref",
]
