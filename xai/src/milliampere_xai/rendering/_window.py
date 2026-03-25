"""Base pygame surface window used by all XAI dashboard panels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pygame
from milliampere_dp.rendering import Color
from milliampere_dp.rendering.pygame_renderer import PygameRenderer


class Window:
    """Base class for pygame surface windows in the XAI dashboard."""

    def __init__(
        self, screen, window_pos, title, window_width, window_height, renderer=None
    ):
        # private attributes
        self._screen = screen
        self._window_pos = window_pos
        self._title = title
        self._title_offset = 5  # pixels
        self._renderer = renderer or PygameRenderer()

        # public attributes
        self.window_width = window_width
        self.window_height = window_height
        self.window_font = pygame.font.SysFont("DejaVu Sans", 11)
        self.antialias = True
        self.surface = pygame.Surface((self.window_width, self.window_height))

    def render_surface(self):
        self.surface.fill(Color.WHITE.value)

    def render_surface_title(self):
        text_surface = self.window_font.render(
            self._title, self.antialias, Color.GRAY.value
        )
        text_rect = text_surface.get_rect()
        text_rect.topright = (
            self.window_width - self._title_offset,
            self._title_offset,
        )
        self.surface.blit(text_surface, text_rect)

    def render_window(self):
        self._screen.blit(self.surface, self._window_pos)

    def draw_arrow(
        self,
        angle,
        arrow_length,
        color,
        start_pos: Any = np.zeros(2),
        arrowhead_length=10,
        arrowhead_width=10,
        line_width=2,
    ):
        if arrow_length < 0:
            arrow_length = -arrow_length
            angle += np.pi

        if arrow_length > arrowhead_length:
            line_length = arrow_length - arrowhead_length
            current_arrowhead_length = arrowhead_length
        else:
            line_length = 0
            current_arrowhead_length = arrow_length
            arrowhead_width *= arrow_length / arrowhead_length

        rotate = np.array([np.cos(angle), np.sin(angle)])

        line_end = start_pos + line_length * rotate
        arrow_tip = line_end + current_arrowhead_length * rotate

        left_base = line_end + arrowhead_width / 2 * np.array([-rotate[1], rotate[0]])
        right_base = line_end - arrowhead_width / 2 * np.array([-rotate[1], rotate[0]])

        if line_length > 0:
            self._renderer.draw_line(
                self.surface, color, start_pos, line_end, line_width
            )
        self._renderer.draw_polygon(
            self.surface, color, np.array([arrow_tip, left_base, right_base])
        )

    def create_legend(
        self,
        legend_items,
        font,
        box_color=Color.LEGEND_BOX.value,
        text_color=Color.BLACK.value,
        marker_size=12,
        spacing=5,
        padding=10,
        margin: float = 10,
        padding_bottom=60,
    ):
        rendered_items = []
        for marker_color, label in legend_items:
            text_surface = font.render(label, self.antialias, text_color)
            rendered_items.append((marker_color, text_surface))

        max_text_width = max(text.get_width() for _, text in rendered_items)
        line_height = max(text.get_height() for _, text in rendered_items)

        total_height = (
            len(rendered_items) * line_height + (len(rendered_items) - 1) * spacing
        )
        box_width = padding * 2 + marker_size + spacing + max_text_width
        box_height = padding * 2 + total_height

        surface_rect = self.surface.get_rect()
        box_rect = pygame.Rect(0, 0, box_width, box_height)
        box_rect.bottomright = (
            int(surface_rect.width - margin),
            int(surface_rect.height - margin - padding_bottom),
        )

        legend_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        legend_surface.fill(box_color)

        y_offset = padding
        for (
            marker_color,
            text_surface,
        ) in rendered_items:
            marker_rect = pygame.Rect(
                padding,
                y_offset + (line_height - marker_size) // 2,
                marker_size,
                marker_size,
            )
            self._renderer.draw_rect(legend_surface, marker_color, marker_rect)

            text_pos = (padding + marker_size + spacing, y_offset)
            legend_surface.blit(text_surface, text_pos)

            y_offset += line_height + spacing

        return legend_surface, box_rect.topleft

    def draw_legend(self, legend_surface, box_pos):
        self.surface.blit(legend_surface, box_pos)
