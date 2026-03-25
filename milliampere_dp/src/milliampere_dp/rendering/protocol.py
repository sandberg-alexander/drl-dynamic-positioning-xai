"""Renderer protocol for vessel visualization.

Uses structural subtyping (PEP 544) so concrete implementations
do not need to inherit.  pyright validates conformance at type-check time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Tuple, Union, runtime_checkable  # noqa: UP035

import numpy as np

try:
    from typing import Protocol
except ImportError:  # Python 3.7
    from typing_extensions import Protocol  # type: ignore[assignment]

# Type alias for color values (RGB or RGBA)
# Use typing.Tuple/Union for Python 3.8 runtime compatibility
ColorType = Union[Tuple[int, int, int], Tuple[int, int, int, int]]  # noqa: UP006, UP007


@runtime_checkable
class Renderer(Protocol):
    """Structural interface for 2D vessel rendering backends.

    Concrete implementations (PygameRenderer, future WebSocketRenderer)
    satisfy this protocol by implementing all methods — no inheritance
    required.  All coordinate parameters accept np.ndarray or plain
    sequences; implementations handle conversion internally.
    """

    def draw_polygon(
        self,
        surface: Any,
        color: ColorType,
        points: np.ndarray | Sequence[Sequence[float]],
    ) -> None:
        """Draw a filled polygon."""
        ...

    def draw_circle(
        self,
        surface: Any,
        color: ColorType,
        center: np.ndarray | Sequence[float],
        radius: float,
        width: int = 0,
    ) -> None:
        """Draw a circle. width=0 means filled, >0 means outline thickness."""
        ...

    def draw_line(
        self,
        surface: Any,
        color: ColorType,
        start: np.ndarray | Sequence[float],
        end: np.ndarray | Sequence[float],
        width: int = 1,
    ) -> None:
        """Draw a line segment."""
        ...

    def draw_lines(
        self,
        surface: Any,
        color: ColorType,
        closed: bool,
        points: Sequence[np.ndarray | Sequence[float]],
        width: int = 1,
    ) -> None:
        """Draw connected line segments."""
        ...

    def draw_rect(
        self,
        surface: Any,
        color: ColorType,
        rect: Any,
    ) -> None:
        """Draw a filled rectangle."""
        ...

    def draw_dashed_line(
        self,
        surface: Any,
        color: ColorType,
        start: np.ndarray | Sequence[float],
        end: np.ndarray | Sequence[float],
        dash_length: float = 10.0,
        gap_length: float = 5.0,
        width: int = 2,
    ) -> None:
        """Draw a dashed line segment."""
        ...
