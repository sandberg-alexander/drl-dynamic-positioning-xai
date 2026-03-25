"""NED-frame position/heading visualization with compass and grid."""

from __future__ import annotations

import numpy as np
from milliampere_dp.rendering import Color

from milliampere_xai.rendering._vessel import VesselRender


class NedRender(VesselRender):
    """NED-frame position/heading visualization with compass and grid."""

    SCALE = 25

    def __init__(
        self,
        screen,
        window_pos,
        title="Global map in NED-frame",
        window_width=450,
        window_height=450,
        renderer=None,
    ):
        super().__init__(
            screen,
            window_pos,
            self.SCALE,
            title,
            window_width,
            window_height,
            renderer=renderer,
        )

    def _draw_grid(
        self, target_pose, spacing: float = 1.0, color=Color.GRAY.value, width: int = 1
    ):
        """
        Draw a world\u2010aligned grid in NED whose lines fall on integer multiples
        of `spacing` (in meters).  The pixel\u2010center of the surface is always
        the world\u2010coordinate (target_pose.x, target_pose.y).
        """
        W, H = self.window_width, self.window_height
        cx, cy = self.center  # pixel where world (tx,ty) sits
        s = self.scale  # pixels per meter
        px = spacing * s  # pixels per grid cell

        # find fractional pixel\u2010offset inside the grid cell
        # i.e. how far (in pixels) our center is into its current meter\u2010grid
        off_x = (target_pose[0] * s) % px
        off_y = (target_pose[1] * s) % px

        # the first vertical line at or to the left of x=0:
        start_x = cx + off_x
        # march left until off\u2010screen, then march right
        x = start_x - ((start_x // px) * px)
        if x > px:
            x -= px

        # similarly for horizontals, but NED wants +Y up \u2192 pixel Y goes down
        start_y = cy - off_y
        y = start_y - ((start_y // px) * px)
        if y > px:
            y -= px

        # draw verticals
        while y < W:
            self._renderer.draw_line(self.surface, color, (y, 0), (y, H), width)
            y += px

        while x < H:
            self._renderer.draw_line(self.surface, color, (0, x), (W, x), width)
            x += px

    def render(self, x_ned_err, y_ned_err, psi_err, target_pose):
        """Render NED map with target at center and vessel offset by NED error.

        Uses the same positioning logic as the standalone viewer:
        vessel position = center - NED error (error = target - vessel).
        """
        self.render_surface()
        self.surface.fill(Color.OCEAN_BLUE.value)
        self._draw_grid(target_pose, spacing=1.0, color=Color.OCEAN_GRID.value, width=1)

        # Target at center, rotated by target heading
        self.draw_target(0, 0, target_pose[2], ned=True)

        # Vessel: offset from center by NED error, rotated by vessel heading
        # error = target - vessel \u2192 vessel pos = -error (from target)
        vessel_heading = target_pose[2] - psi_err
        self._draw_vessel_ned(-x_ned_err, -y_ned_err, vessel_heading)

        self.render_surface_title()
        self.render_window()

    def _draw_vessel_ned(self, world_x, world_y, heading_deg):
        """Draw vessel at world position (NED meters from target) with given heading."""
        angle = np.deg2rad(heading_deg)
        # Rotate vessel shape by heading, then translate to world position
        R_rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        shape = (R_rot @ self.shape.T).T + np.array([world_x, world_y])
        triangle = (R_rot @ self.triangle_shape.T).T + np.array([world_x, world_y])
        circle = (R_rot @ self.circle_point.T).T + np.array([world_x, world_y])

        # Convert world \u2192 pixels and draw
        self.vessel_surface.fill(Color.AGENT_BLACK.value)
        self._renderer.draw_polygon(
            self.vessel_surface, Color.AGENT_BLUE.value, self._world_2_pixels(shape)
        )
        self._renderer.draw_polygon(
            self.vessel_surface, Color.BLACK.value, self._world_2_pixels(triangle)
        )
        self._renderer.draw_circle(
            self.vessel_surface,
            Color.BLACK.value,
            self._world_2_pixels(circle).ravel(),
            radius=3,
        )
        self.surface.blit(self.vessel_surface, (0, 0))
