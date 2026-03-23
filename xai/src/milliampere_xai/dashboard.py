"""Deprecated: use milliampere_xai.rendering instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "milliampere_xai.dashboard is deprecated; use milliampere_xai.rendering",
    DeprecationWarning,
    stacklevel=2,
)

from milliampere_xai.rendering._compositor import RenderExplanation  # noqa: E402, F401

__all__ = ["RenderExplanation"]
