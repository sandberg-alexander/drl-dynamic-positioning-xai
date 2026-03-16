"""Shared fixtures for milliampere_env tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src directory is importable without catkin
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "env"


@pytest.fixture()
def configs_dir() -> Path:
    return CONFIGS_DIR
