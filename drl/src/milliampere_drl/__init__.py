"""milliampere_drl -- DRL training, evaluation, and deployment for milliAmpere1 DP."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("milliampere-drl")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
