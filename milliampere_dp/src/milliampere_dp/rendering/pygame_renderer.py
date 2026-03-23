"""Pygame-based renderer for vessel visualization.

Lazy-imports pygame so that milliampere_dp can be imported without
pygame installed.  Only fails when PygameRenderer is instantiated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from milliampere_dp.rendering.protocol import ColorType


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
        self._pygame.draw.polygon(surface, color, points)  # type: ignore[arg-type]

    def draw_circle(
        self,
        surface: Any,
        color: ColorType,
        center: Sequence[float],
        radius: float,
    ) -> None:
        self._pygame.draw.circle(surface, color, center, radius)

    def draw_line(
        self,
        surface: Any,
        color: ColorType,
        start: Sequence[float],
        end: Sequence[float],
        width: int = 1,
    ) -> None:
        self._pygame.draw.line(surface, color, start, end, width)

    def draw_lines(
        self,
        surface: Any,
        color: ColorType,
        closed: bool,
        points: Sequence[Sequence[float]],
        width: int = 1,
    ) -> None:
        self._pygame.draw.lines(surface, color, closed, points, width)

    def draw_dashed_line(
        self,
        surface: Any,
        color: ColorType,
        start: Sequence[float],
        end: Sequence[float],
        dash_length: float = 10.0,
        gap_length: float = 5.0,
        width: int = 2,
    ) -> None:
        """Consolidated dashed-line drawing (fixes dashboard ``dash_lenght`` typo)."""
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
