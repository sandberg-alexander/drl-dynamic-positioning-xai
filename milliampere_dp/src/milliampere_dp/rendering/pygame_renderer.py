"""Pygame-based renderer for vessel visualization.

Lazy-imports pygame so that milliampere_dp can be imported without
pygame installed.  Only fails when PygameRenderer is instantiated.

All methods accept np.ndarray for coordinates and convert to
list/tuple internally before passing to pygame.draw.*.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from milliampere_dp.rendering.protocol import ColorType


def _to_coords(val: Any) -> Any:
    """Convert ndarray to list for pygame compatibility."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _to_point(val: Any) -> tuple[float, ...]:
    """Convert ndarray or sequence to a flat tuple for pygame."""
    if isinstance(val, np.ndarray):
        return tuple(val.ravel().tolist())
    return tuple(val)


class PygameRenderer:
    """Pygame implementation of the Renderer protocol.

    Does NOT inherit from Renderer — structural subtyping via Protocol.
    Surface parameters are typed as ``Any`` to avoid importing pygame
    at module level (lazy import pattern).
    """

    def __init__(self) -> None:
        import pygame as _pygame

        self._pygame = _pygame

    def draw_polygon(
        self,
        surface: Any,
        color: ColorType,
        points: np.ndarray | Sequence[Sequence[float]],
    ) -> None:
        self._pygame.draw.polygon(surface, color, _to_coords(points))

    def draw_circle(
        self,
        surface: Any,
        color: ColorType,
        center: np.ndarray | Sequence[float],
        radius: float,
        width: int = 0,
    ) -> None:
        self._pygame.draw.circle(surface, color, _to_point(center), radius, width)

    def draw_line(
        self,
        surface: Any,
        color: ColorType,
        start: np.ndarray | Sequence[float],
        end: np.ndarray | Sequence[float],
        width: int = 1,
    ) -> None:
        self._pygame.draw.line(surface, color, _to_point(start), _to_point(end), width)

    def draw_lines(
        self,
        surface: Any,
        color: ColorType,
        closed: bool,
        points: Sequence[np.ndarray | Sequence[float]],
        width: int = 1,
    ) -> None:
        converted = [_to_point(p) for p in points]
        self._pygame.draw.lines(surface, color, closed, converted, width)

    def draw_rect(
        self,
        surface: Any,
        color: ColorType,
        rect: Any,
    ) -> None:
        self._pygame.draw.rect(surface, color, rect)

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
        """Consolidated dashed-line drawing."""
        px1, py1 = float(start[0]), float(start[1])
        px2, py2 = float(end[0]), float(end[1])
        length = math.hypot(px2 - px1, py2 - py1)
        if length == 0:
            return
        dashes = int(length / (dash_length + gap_length))
        if dashes == 0:
            return
        for i in range(dashes):
            t1 = i / dashes
            t2 = (i + 0.5) / dashes
            seg_start = (px1 + (px2 - px1) * t1, py1 + (py2 - py1) * t1)
            seg_end = (px1 + (px2 - px1) * t2, py1 + (py2 - py1) * t2)
            self._pygame.draw.line(surface, color, seg_start, seg_end, width)
