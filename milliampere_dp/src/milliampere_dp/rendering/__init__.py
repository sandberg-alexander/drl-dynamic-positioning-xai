"""Rendering primitives for milliAmpere1 vessel visualization.

Provides coordinate transforms, vessel geometry, color constants,
a Renderer Protocol, and Pydantic frame schemas -- all decoupled
from any specific rendering backend (pygame, matplotlib, etc.).
"""

from __future__ import annotations

from milliampere_dp.rendering.colors import Color
from milliampere_dp.rendering.geometry import (
    VESSEL_CORNER,
    VESSEL_MOMENT_MARKER,
    VESSEL_TRIANGLE,
    vessel_hull_polygon,
    vessel_moment_marker_line,
    vessel_triangle_polygon,
)
from milliampere_dp.rendering.protocol import ColorType, Renderer
from milliampere_dp.rendering.transforms import (
    R2,
    T2,
    Viewport,
    degrees_to_pygame,
    meters_to_pixels,
    transform_body_to_ned,
    transform_shape,
)

__all__ = [
    # colors
    "Color",
    # geometry
    "VESSEL_CORNER",
    "VESSEL_TRIANGLE",
    "VESSEL_MOMENT_MARKER",
    "vessel_hull_polygon",
    "vessel_triangle_polygon",
    "vessel_moment_marker_line",
    # transforms
    "Viewport",
    "R2",
    "T2",
    "transform_shape",
    "transform_body_to_ned",
    "degrees_to_pygame",
    "meters_to_pixels",
    # protocol
    "Renderer",
    "ColorType",
]
