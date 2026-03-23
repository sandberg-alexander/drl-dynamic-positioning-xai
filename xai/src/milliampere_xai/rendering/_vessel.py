"""Vessel outline drawing with thrusters, body geometry, and coordinate transforms."""

from __future__ import annotations

import math

import numpy as np
import pygame
from milliampere_dp.rendering import Color
from milliampere_dp.rendering.geometry import (
    vessel_hull_polygon,
    vessel_moment_marker_line,
    vessel_triangle_polygon,
)
from milliampere_dp.rendering.transforms import (
    R2,
    T2,
    Viewport,
    degrees_to_pygame,
    meters_to_pixels,
    transform_body_to_ned,
    transform_shape,
)
from milliampere_dp.vessel import VESSEL_BEAM, VESSEL_LENGTH

from milliampere_xai.rendering._window import Window


class VesselRender(Window):
    """Base class for vessel outline drawing with thrusters and body geometry."""

    VESSEL_LENGTH = VESSEL_LENGTH  # from milliampere_dp.vessel (5.06 m)
    VESSEL_WIDTH = VESSEL_BEAM  # from milliampere_dp.vessel (2.86 m)
    VESSEL_CORNER = 0.5  # meters
    VESSEL_TRIANGLE = 0.4  # meters
    VESSEL_MOMENT_MARKER = 0.6  # meters

    def __init__(self, screen, window_pos, scale, title, window_width, window_height):
        super().__init__(screen, window_pos, title, window_width, window_height)

        self._viewport = Viewport.from_window(scale, window_width, window_height)
        self.scale = self._viewport.scale
        self.center = self._viewport.center
        self.R = self._viewport.R
        self.T = self._viewport.T

        self.vessel_surface = pygame.Surface(
            (window_width, window_height), pygame.SRCALPHA
        )
        self.shape = vessel_hull_polygon()
        self.circle_point = np.zeros((1, 2))
        self.triangle_shape = vessel_triangle_polygon()
        self.moment_marker_line = vessel_moment_marker_line()

    def _draw_body_grid(
        self,
        x_err,
        y_err,
        psi_err,
        target_pose,
        spacing: float = 1.0,
        color=Color.GRAY.value,
        width: int = 1,
    ):
        """
        Same pinning logic as before, but with the grid rotated by `psi_err`
        degrees (positive CCW).  Everything else is unchanged.
        """
        # ------------------------------------------------------------------
        # 0.  Pre-compute a few things we need again and again
        # ------------------------------------------------------------------
        circle = self._transform_vessel(x_err, y_err, psi_err, self.circle_point)
        W, H = self.window_width, self.window_height
        cx, cy = self._world_2_pixels(circle).ravel()  # vessel-pixel centre

        s = self.scale  # px / metre
        px = spacing * s  # px per grid cell

        # where (x_n, y_n) is the pixel that represents target_pose[:2]
        x_n, y_n = self._world_2_pixels(np.array([list(target_pose[:2])])).ravel()

        x_p = x_n - 25 - int(x_n / px) * px
        y_p = y_n - 25 - int(y_n / px) * px

        # ------------------------------------------------------------------
        # 1.  Build the list of UNROTATED line offsets  (exactly what you had)
        # ------------------------------------------------------------------
        horiz = []
        vert = []

        # horizontal lines --------------------------------------------------
        y0 = cy - y_p
        horiz.append(y0)
        dy = px
        while y0 + dy <= H:
            horiz.append(y0 + dy)
            dy += px
        dy = px
        while y0 - dy >= 0:
            horiz.append(y0 - dy)
            dy += px

        # vertical lines ----------------------------------------------------
        x0 = cx - x_p
        vert.append(x0)
        dx = px
        while x0 + dx <= W:
            vert.append(x0 + dx)
            dx += px
        dx = px
        while x0 - dx >= 0:
            vert.append(x0 - dx)
            dx += px

        # ------------------------------------------------------------------
        # 2.  Rotate every line about (cx, cy) by  \u03c8 = psi_err  degrees
        # ------------------------------------------------------------------
        ang = math.radians(psi_err - target_pose[2])  # positive = CCW
        cosA = math.cos(ang)
        sinA = math.sin(ang)

        # screen diagonal \u2013 long enough so that rotated lines cover the view
        diag = math.hypot(W, H)

        # helper: rotate a point round the centre
        def _rot(pt):
            x, y = pt
            dx, dy = x - cx, y - cy  # translate to origin
            xr = dx * cosA - dy * sinA
            yr = dx * sinA + dy * cosA
            return (cx + xr, cy + yr)

        # draw horizontal (world-Y) lines  ----------------------------------
        for y in horiz:
            p1 = _rot((-diag, y))  # far left
            p2 = _rot((diag, y))  # far right
            pygame.draw.line(self.surface, color, p1, p2, width)

        # draw vertical (world-X) lines  ------------------------------------
        for x in vert:
            p1 = _rot((x, -diag))  # far up
            p2 = _rot((x, diag))  # far down
            pygame.draw.line(self.surface, color, p1, p2, width)

    def draw_vessel(self, x_err=None, y_err=None, psi_err=None, ned=False):
        if ned:
            shape = self._transform_body2ned(x_err, y_err, psi_err, self.shape)
            circle = self._transform_body2ned(x_err, y_err, psi_err, self.circle_point)
            triangle = self._transform_body2ned(
                x_err, y_err, psi_err, self.triangle_shape
            )
        else:
            shape = self.shape
            circle = self.circle_point
            triangle = self.triangle_shape

        self.vessel_surface.fill(Color.AGENT_BLACK.value)
        pygame.draw.polygon(
            self.vessel_surface, Color.AGENT_BLUE.value, self._world_2_pixels(shape)
        )
        pygame.draw.polygon(
            self.vessel_surface, Color.BLACK.value, self._world_2_pixels(triangle)
        )
        pygame.draw.circle(
            self.vessel_surface,
            Color.BLACK.value,
            self._world_2_pixels(circle).ravel(),
            radius=3,
        )
        if not ned:
            line = self._world_2_pixels(self.moment_marker_line)
            pygame.draw.line(self.vessel_surface, Color.BLACK.value, line[0], line[1])
        self.surface.blit(self.vessel_surface, (0, 0))

    def draw_target(
        self, x_err, y_err, psi_err, ned=False, color=Color.DESIRED_YELLOW.value
    ):
        if ned:
            shape = self._transform_vessel(0, 0, psi_err, self.shape)
            # shape = self.shape
            circle = self.circle_point
            # triangle = self.triangle_shape
            triangle = self._transform_vessel(0, 0, psi_err, self.triangle_shape)

        else:
            shape = self._transform_vessel(x_err, y_err, psi_err, self.shape)
            circle = self._transform_vessel(x_err, y_err, psi_err, self.circle_point)
            triangle = self._transform_vessel(
                x_err, y_err, psi_err, self.triangle_shape
            )

        points = self._world_2_pixels(shape)
        triangle_points = self._world_2_pixels(triangle)

        for i in range(len(shape)):
            self._draw_dashed_line(points[i], points[(i + 1) % len(points)], color)
            if i < len(triangle):
                self._draw_dashed_line(
                    triangle_points[i],
                    triangle_points[(i + 1) % len(triangle_points)],
                    color,
                    dash_lenght=5,
                )

        # Polygon outline removed — using dashed lines instead
        pygame.draw.circle(
            self.surface, color, self._world_2_pixels(circle).ravel(), radius=3
        )

    def _draw_dashed_line(
        self, start_pos, end_pos, color, dash_lenght=10, gap_length=5
    ):
        px1, py1 = start_pos
        px2, py2 = end_pos
        length = ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5
        dashes = int(length / (dash_lenght + gap_length))
        for i in range(dashes):
            t1 = i / dashes
            t2 = (i + 0.5) / dashes
            segment_start = (px1 + (px2 - px1) * t1, py1 + (py2 - py1) * t1)
            segment_end = (px1 + (px2 - px1) * t2, py1 + (py2 - py1) * t2)
            pygame.draw.line(self.surface, color, segment_start, segment_end, 2)

    # Transformations \u2014 thin wrappers delegating to milliampere_dp.rendering

    def _transform_vessel(self, x, y, psi, shape):
        return transform_shape(x, y, psi, shape)

    def _transform_body2ned(self, x, y, psi, shape):
        return transform_body_to_ned(x, y, psi, shape)

    def _R2(self, psi):
        return R2(psi)

    def _T2(self, x, y):
        return T2(x, y)

    def _world_2_pixels(self, cord):
        return self._viewport.world_to_pixels(cord)

    def _degrees2pygame(self, angle):
        return degrees_to_pygame(angle)

    def _scalar2pygame(self, scalar):
        return meters_to_pixels(self.scale, scalar)
