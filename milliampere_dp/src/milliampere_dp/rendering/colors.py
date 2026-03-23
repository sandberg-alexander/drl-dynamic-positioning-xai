"""Color constants for milliAmpere1 vessel rendering.

Extracted from dashboard.py Color enum. RGB tuples for opaque colors,
RGBA tuples where alpha is needed.
"""

from __future__ import annotations

from enum import Enum


class Color(Enum):
    """Color constants for vessel rendering."""

    WHITE = (255, 255, 255)
    GRAY = (150, 150, 150)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    PURPLE = (255, 0, 255)
    RED = (255, 0, 0)
    ORANGE = (255, 128, 0)

    SCREEN_COLOR = (240, 240, 240)

    TABLEAU_BLUE = (87, 120, 164)
    TABLEAU_ORANGE = (228, 148, 68)
    TABLEAU_GREEN = (106, 159, 88)
    TABLEAU_RED = (209, 97, 93)

    LEGEND_BOX = (200, 200, 200, 180)

    AGENT_BLACK = (0, 0, 0, 0)
    AGENT_BLUE = (0, 122, 255, 220)
    DESIRED_YELLOW = (255, 102, 0)
    DESIRED_LIGHT_YELLOW = (255, 205, 128)
    OCEAN_BLUE = (209, 237, 255)
    OCEAN_GRID = (182, 206, 222)
    VELOCITY_GREEN = (0, 255, 123)

    LIGHT_RED = (255, 171, 171)

    @property
    def hex(self) -> str:
        """Web-compatible hex color string (e.g. '#ff0000').

        Uses only the first 3 channels (ignores alpha for RGBA colors).
        """
        r, g, b = self.value[0], self.value[1], self.value[2]
        return f"#{r:02x}{g:02x}{b:02x}"
