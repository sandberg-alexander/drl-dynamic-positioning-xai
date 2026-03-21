"""Shared fixtures for integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure milliampere_env is importable without catkin
_ENV_SRC = (
    Path(__file__).resolve().parent.parent.parent
    / "ros_packages"
    / "milliampere_env"
    / "src"
)
if str(_ENV_SRC) not in sys.path:
    sys.path.insert(0, str(_ENV_SRC))

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "env"


@pytest.fixture()
def configs_dir() -> Path:
    return CONFIGS_DIR
