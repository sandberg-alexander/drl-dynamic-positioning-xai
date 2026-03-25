"""Shared drawing primitives: arrows, arcs, distance markers, and angle indicators."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pygame

if TYPE_CHECKING:
    from milliampere_dp.rendering.protocol import Renderer


class Utilities:
    """Shared drawing primitives: arrows, arcs, distance/angle markers.

    This is a mixin class — expects ``self.surface`` and ``self._renderer``
    to be provided by the concrete class (via Window through MRO).
    """

    # Declare attributes provided by Window via MRO so pyright knows about them
    surface: Any
    _renderer: Renderer

    def __init__(self):
        pass

    def _draw_arrow(
        self,
        angle,
        arrow_length,
        color,
        start_pos: Any = np.zeros(2),
        arrowhead_length=10,
        arrowhead_width=10,
        line_width=2,
        show_measurement=False,
        label=None,
        font=None,
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
            self.surface, color, [arrow_tip, left_base, right_base]
        )

        # Display measurement text
        if show_measurement:
            if font is None:
                font = pygame.font.SysFont("Arial", 18)

            text = label
            text_surface = font.render(text, True, color)

            # Position text above the middle of the line
            text_pos_x = (
                start_pos[0] + arrow_tip[0]
            ) / 2 - text_surface.get_width() / 2
            text_pos_y = (
                (start_pos[1] + arrow_tip[1]) / 2 - text_surface.get_height() - 5
            )

            # Adjust text position if line is vertical
            if np.pi / 2 - np.pi / 16 < abs(angle) < np.pi / 2 + np.pi / 16:
                text_pos_x = (start_pos[0] + arrow_tip[0]) / 2 + 10
                text_pos_y = (
                    start_pos[1] + arrow_tip[1]
                ) / 2 - text_surface.get_height() / 2

            self.surface.blit(text_surface, (text_pos_x, text_pos_y))

    def _draw_curved_arrow(
        self,
        angle,
        center,
        color,
        radius: float = 10,
        line_width=2,
        arrowhead_length=10,
        arrowhead_width=10,
        num_points=30,
        show_measurement=False,
        font=None,
        label=None,
    ):
        if angle == 0:
            return

        sign = 1 if angle > 0 else -1

        total_arc_length = radius * abs(angle)

        if total_arc_length > arrowhead_length:
            line_arc_length = total_arc_length - arrowhead_length
            current_arrowhead_length = arrowhead_length
        else:
            line_arc_length = 0
            current_arrowhead_length = total_arc_length
            arrowhead_width *= total_arc_length / arrowhead_length

        line_arc_angle = line_arc_length / radius
        arrowhead_arc_angle = current_arrowhead_length / radius

        start_angle = 3 * np.pi / 2
        tip_angle = start_angle + sign * (line_arc_angle + arrowhead_arc_angle)

        if line_arc_length > 0:
            arc_points: list[tuple[float, float]] = []
            for i in range(num_points):
                t = i / (num_points - 1)
                theta = start_angle + sign * (line_arc_angle * t)
                pos = center + radius * np.array([np.cos(theta), np.sin(theta)])
                arc_points.append(tuple(pos))
            self._renderer.draw_lines(
                self.surface, color, False, arc_points, line_width
            )

        rotate = np.array([np.cos(tip_angle), np.sin(tip_angle)])
        tip = center + radius * rotate
        base_center = tip - current_arrowhead_length * sign * np.array(
            [-rotate[1], rotate[0]]
        )
        left_corner = base_center + arrowhead_width / 2 * sign * rotate
        right_corner = base_center - arrowhead_width / 2 * sign * rotate

        self._renderer.draw_polygon(
            self.surface, color, [tip, left_corner, right_corner]
        )

        # Display measurement text if needed
        if show_measurement:
            if font is None:
                font = pygame.font.SysFont("Arial", 18)

            # Create the label text
            if label is None:
                angle_degrees = np.degrees(abs(angle))
                label = f"{angle_degrees:.1f}\u00b0"

            text_surface = font.render(label, True, color)

            # Position text in the middle of the arc
            mid_angle = start_angle + sign * (angle / 2)
            if sign == 1:
                mid_vector = np.array([np.cos(mid_angle), np.sin(mid_angle)])
            else:
                mid_vector = np.array([-np.cos(mid_angle), np.sin(mid_angle)])
            text_pos = (
                center + radius * 1.2 * mid_vector
            )  # Position slightly outside the arc

            # Offset to center the text
            text_rect = text_surface.get_rect()
            text_rect.center = text_pos

            self.surface.blit(text_surface, text_rect)

    def _draw_distance(
        self,
        surface,
        start_pos,
        length,
        angle_rad,
        color=(255, 255, 255),
        line_width=2,
        perpendicular_length=10,
        font=None,
        show_measurement=True,
        units="px",
        label=None,
    ):
        """
        Draw a distance measurement line with perpendicular markers at each end.

        Args:
            surface: Pygame surface to draw on
            start_pos: (x, y) tuple of the starting position
            length: Length of the measurement line
            angle_degrees: Angle of the line in degrees (0 = right, 90 = down)
            color: RGB tuple for line color
            line_width: Width of the line in pixels
            perpendicular_length: Length of the perpendicular end markers
            font: Pygame font object for text (if None, default font will be used)
            show_measurement: Whether to display the measurement text
            units: String for measurement units
        """
        # Convert angle to radians
        # angle_rad = math.radians(angle_degrees)

        # Calculate end position
        end_x = start_pos[0] + length * math.cos(angle_rad)
        end_y = start_pos[1] + length * math.sin(angle_rad)
        end_pos = (end_x, end_y)

        # Draw the main line
        self._renderer.draw_line(surface, color, start_pos, end_pos, line_width)

        # Calculate perpendicular angle
        perp_angle = angle_rad + math.pi / 2  # 90 degrees in radians

        # Draw perpendicular line at start position
        perp_start_x1 = start_pos[0] + perpendicular_length / 2 * math.cos(perp_angle)
        perp_start_y1 = start_pos[1] + perpendicular_length / 2 * math.sin(perp_angle)
        perp_start_x2 = start_pos[0] - perpendicular_length / 2 * math.cos(perp_angle)
        perp_start_y2 = start_pos[1] - perpendicular_length / 2 * math.sin(perp_angle)
        self._renderer.draw_line(
            surface,
            color,
            (perp_start_x1, perp_start_y1),
            (perp_start_x2, perp_start_y2),
            line_width,
        )

        # Draw perpendicular line at end position
        perp_end_x1 = end_pos[0] + perpendicular_length / 2 * math.cos(perp_angle)
        perp_end_y1 = end_pos[1] + perpendicular_length / 2 * math.sin(perp_angle)
        perp_end_x2 = end_pos[0] - perpendicular_length / 2 * math.cos(perp_angle)
        perp_end_y2 = end_pos[1] - perpendicular_length / 2 * math.sin(perp_angle)
        self._renderer.draw_line(
            surface,
            color,
            (perp_end_x1, perp_end_y1),
            (perp_end_x2, perp_end_y2),
            line_width,
        )

        # Display measurement text
        if show_measurement:
            if font is None:
                font = pygame.font.SysFont("Arial", 18)

            text = label
            text_surface = font.render(text, True, color)

            # Position text above the middle of the line
            text_pos_x = (start_pos[0] + end_pos[0]) / 2 - text_surface.get_width() / 2
            text_pos_y = (
                start_pos[1] + end_pos[1]
            ) / 2 + 5  # - text_surface.get_height() - 5

            # Adjust text position if line is vertical
            if np.pi / 2 - np.pi / 16 < abs(angle_rad) < np.pi / 2 + np.pi / 16:
                text_pos_x = (start_pos[0] + end_pos[0]) / 2 + 10
                text_pos_y = (
                    start_pos[1] + end_pos[1]
                ) / 2 - text_surface.get_height() / 2

            surface.blit(text_surface, (text_pos_x, text_pos_y))

    def _draw_angle(
        self,
        surface,
        center,
        angle,
        color=(255, 255, 255),
        radius: float = 30,
        line_width=2,
        perpendicular_length=10,
        font=None,
        show_measurement=True,
        label=None,
        num_points=30,
        start_angle=3 * np.pi / 2,
    ):
        """
        Draw an angle measurement with an arc and perpendicular markers at endpoints.
        Args:
            surface: Pygame surface to draw on
            center: (x, y) numpy array of the center point
            angle: Angle in radians (positive = CCW, negative = CW)
            color: RGB tuple for line color
            radius: Radius of the arc
            line_width: Width of the line in pixels
            perpendicular_length: Length of the perpendicular end markers
            font: Pygame font object for text
            show_measurement: Whether to display the measurement text
            label: Custom label text
            num_points: Number of points for drawing the arc
            start_angle: Starting angle in radians (default 3*pi/2 = up)
        """
        # Handle zero angle case
        if angle == 0:
            return

        # Determine direction based on angle sign
        sign = 1 if angle > 0 else -1

        # Calculate the arc angle
        arc_angle = abs(angle)

        # Calculate end angle
        end_angle = start_angle + sign * arc_angle

        # Draw the arc
        arc_points: list[tuple[float, float]] = []
        for i in range(num_points):
            t = i / (num_points - 1)
            theta = start_angle + sign * (arc_angle * t)
            pos = center + radius * np.array([np.cos(theta), np.sin(theta)])
            arc_points.append(tuple(pos))

        self._renderer.draw_lines(surface, color, False, arc_points, line_width)

        # Draw perpendicular markers at the endpoints
        # Start point - use radial direction (from center to point)
        start_vector = np.array([np.cos(start_angle), np.sin(start_angle)])
        start_point = center + radius * start_vector

        # For perpendicular to the arc, use the radial vector itself
        start_inner = start_point - (perpendicular_length / 2) * start_vector
        start_outer = start_point + (perpendicular_length / 2) * start_vector

        self._renderer.draw_line(surface, color, start_inner, start_outer, line_width)

        # End point - use radial direction
        end_vector = np.array([np.cos(end_angle), np.sin(end_angle)])
        end_point = center + radius * end_vector

        # For perpendicular to the arc, use the radial vector itself
        end_inner = end_point - (perpendicular_length / 2) * end_vector
        end_outer = end_point + (perpendicular_length / 2) * end_vector

        self._renderer.draw_line(surface, color, end_inner, end_outer, line_width)

        # Display measurement text if needed
        if show_measurement:
            if font is None:
                font = pygame.font.SysFont("Arial", 18)

            # Create the label text
            if label is None:
                angle_degrees = np.degrees(abs(angle))
                label = f"{angle_degrees:.1f}\u00b0"

            text_surface = font.render(label, True, color)

            # Position text in the middle of the arc
            mid_angle = start_angle + sign * (arc_angle / 2)
            mid_vector = np.array([np.cos(mid_angle), np.sin(mid_angle)])
            text_pos = (
                center + radius * 1.2 * mid_vector
            )  # Position slightly outside the arc

            # Offset to center the text
            text_rect = text_surface.get_rect()
            text_rect.center = text_pos

            surface.blit(text_surface, text_rect)
