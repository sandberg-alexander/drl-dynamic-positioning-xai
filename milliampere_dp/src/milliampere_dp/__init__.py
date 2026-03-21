"""milliampere_dp -- shared domain library for milliAmpere1 dynamic positioning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("milliampere-dp")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
