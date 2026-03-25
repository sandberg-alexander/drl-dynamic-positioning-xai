"""Body-frame thruster visualization with actuator state and velocity arrows."""

from __future__ import annotations

import math

import numpy as np
import pygame
from milliampere_dp.rendering import Color
from milliampere_dp.vessel import THRUSTER_ARM_X, THRUSTER_ARM_Y

from milliampere_xai.rendering._vessel import VesselRender


class BodyRender(VesselRender):
    """Body-frame thruster visualization with actuator state and velocity arrows."""

    SCALE = 50
    ACTUATOR_X = THRUSTER_ARM_X  # from milliampere_dp.vessel (1.8 m)
    ACTUATOR_Y = THRUSTER_ARM_Y  # from milliampere_dp.vessel (0.8 m)

    def __init__(
        self,
        screen,
        window_pos,
        title="State and desired actuator view in BODY-frame",
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

        self.label_font = pygame.font.SysFont("DejaVu Sans", 12)
        self.body_legend_items = (
            (
                Color.DESIRED_YELLOW.value,
                "Desired {'total thrust force', 'total trust moment', 'pose'}",
            ),
            (Color.DESIRED_LIGHT_YELLOW.value, "Desired thrust force"),
            (Color.VELOCITY_GREEN.value, "{'surge', 'sway', 'angular'} velocity"),
        )

        self.legend_surface, self.box_pos = self.create_legend(
            self.body_legend_items,
            self.label_font,
            spacing=5,
            padding=5,
            margin=5,
            padding_bottom=0,
        )

        self.actuator_pos = np.array(
            [
                [self.ACTUATOR_X, -self.ACTUATOR_Y],
                [self.ACTUATOR_X, self.ACTUATOR_Y],
                [-self.ACTUATOR_X, self.ACTUATOR_Y],
                [-self.ACTUATOR_X, -self.ACTUATOR_Y],
            ]
        )

        self._make_compass_base(radius=30)
        cx = cy = self._compass_base.get_width() // 2
        self._compass_screen_center = (10 + cx, 10 + cy)

    def _draw_actuator_ref(self, actuator_ref):
        for i, (x, y) in enumerate(self.actuator_pos):
            px, py = self._world_2_pixels(np.array([[x, y]])).ravel()
            angle = self._degrees2pygame(actuator_ref[i][1])
            perp_angle = angle + np.pi / 2
            self.draw_arrow(
                self._degrees2pygame(actuator_ref[i][1]),
                self._scalar2pygame(actuator_ref[i][0]),
                Color.DESIRED_LIGHT_YELLOW.value,
                np.array([px, py]),
            )
            self._renderer.draw_line(
                self.surface,
                Color.DESIRED_LIGHT_YELLOW.value,
                (px - 10 * np.cos(perp_angle), py - 10 * np.sin(perp_angle)),
                (px + 10 * np.cos(perp_angle), py + 10 * np.sin(perp_angle)),
                width=2,
            )
            self._renderer.draw_circle(
                self.surface, Color.BLACK.value, (px, py), radius=3
            )
            self._renderer.draw_circle(
                self.surface,
                Color.BLACK.value,
                (px - 7 * np.sin(perp_angle), py + 7 * np.cos(perp_angle)),
                radius=1,
            )

    def _draw_force_and_moment(self, n_d, alpha_d, angular_thrust_ref):
        center = self._world_2_pixels(np.zeros((1, 2))).ravel()
        self.draw_arrow(
            self._degrees2pygame(alpha_d),
            self._scalar2pygame(n_d),
            Color.DESIRED_YELLOW.value,
            center,
        )
        radius = self._scalar2pygame(self.VESSEL_MOMENT_MARKER)
        angular_thrust_ref1 = angular_thrust_ref * self.SCALE / radius
        self._draw_curved_arrow(
            angular_thrust_ref1, center, Color.DESIRED_YELLOW.value, radius
        )

    def _draw_velocities(self, u_hat, v_hat, r_hat):
        center = self._world_2_pixels(np.zeros((1, 2))).ravel()
        self.draw_arrow(
            self._degrees2pygame(0),
            self._scalar2pygame(u_hat),
            Color.VELOCITY_GREEN.value,
            center,
        )
        self.draw_arrow(
            self._degrees2pygame(90),
            self._scalar2pygame(v_hat),
            Color.VELOCITY_GREEN.value,
            center,
        )
        self._draw_curved_arrow(
            np.deg2rad(r_hat),
            center,
            Color.VELOCITY_GREEN.value,
            self._scalar2pygame(self.VESSEL_MOMENT_MARKER),
        )

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

    def calculate_total_moment(self, thrusters):
        total_moment_prime = 0.0
        for i, ((x, y), (n, alpha)) in enumerate(zip(self.actuator_pos, thrusters)):
            rad = np.deg2rad(90 * (i + 1) - alpha)
            if x * y < 0:
                M_prime = abs(x) * n * np.cos(rad) + abs(y) * n * np.sin(rad)
            else:
                M_prime = abs(x) * n * np.sin(rad) + abs(y) * n * np.cos(rad)
            total_moment_prime += M_prime
        return total_moment_prime

    def _make_compass_base(self, radius=50):
        """
        Build a Surface that contains:
          \u2022 circle + four triangular points
          \u2022 N/E/S/W letters, positioned *beyond* the circle
        This surface is padded so that rotating it never clips the letters.
        """
        # how far beyond the circle to put each letter:
        letter_offset = 15
        # total half-size of the surf:
        half = radius + letter_offset + 10
        size = half * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx = cy = size / 2
        col = Color.OCEAN_GRID.value

        # draw circle
        self._renderer.draw_circle(surf, col, (cx, cy), radius, width=2)

        # draw the four little triangles (same as before)
        tri_h, tri_w = 30, 8
        # N
        self._renderer.draw_polygon(
            surf,
            col,
            [
                (cx, cy - radius),
                (cx + tri_w, cy - radius + tri_h),
                (cx - tri_w, cy - radius + tri_h),
            ],
        )
        # E
        self._renderer.draw_polygon(
            surf,
            col,
            [
                (cx + radius, cy),
                (cx + radius - tri_h, cy + tri_w),
                (cx + radius - tri_h, cy - tri_w),
            ],
        )
        # S
        self._renderer.draw_polygon(
            surf,
            col,
            [
                (cx, cy + radius),
                (cx + tri_w, cy + radius - tri_h),
                (cx - tri_w, cy + radius - tri_h),
            ],
        )
        # W
        self._renderer.draw_polygon(
            surf,
            col,
            [
                (cx - radius, cy),
                (cx - radius + tri_h, cy + tri_w),
                (cx - radius + tri_h, cy - tri_w),
            ],
        )

        # now *also* draw N/E/S/W letters *on this same surf*
        font = pygame.font.SysFont("DejaVu Sans", 16, bold=True)
        for letter, deg in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
            txt = font.render(letter, True, col)
            w, h = txt.get_size()
            # angle for placement: rotate so that 0\u00b0 is upward
            a = math.radians(deg - 90)
            x = cx + math.cos(a) * (radius + letter_offset) - w / 2
            y = cy + math.sin(a) * (radius + letter_offset) - h / 2
            surf.blit(txt, (x, y))

        self._compass_base = surf
        self._compass_radius = radius

    def draw_compass(self, heading_deg):
        """
        Rotate the *entire* pre-made compass (circle+triangles+letters)
        and blit it so its CENTER stays at _compass_screen_center.
        """
        rose = pygame.transform.rotate(self._compass_base, -heading_deg)
        rect = rose.get_rect(center=self._compass_screen_center)
        self.surface.blit(rose, rect)

    def render(
        self,
        actuator_ref,
        tot_thrust,
        tot_angle,
        tot_angular_thrust,
        x_tilde,
        y_tilde,
        psi_tilde,
        u_hat,
        v_hat,
        r_hat,
        target_pose,
    ):
        # init
        self.render_surface()

        # BODY render
        self.surface.fill(Color.OCEAN_BLUE.value)

        self._draw_body_grid(
            x_tilde,
            y_tilde,
            psi_tilde,
            target_pose,
            spacing=1.0,
            color=Color.OCEAN_GRID.value,
            width=1,
        )
        self.draw_target(x_tilde, y_tilde, psi_tilde)
        self.draw_vessel(x_tilde, y_tilde, psi_tilde)
        self._draw_actuator_ref(actuator_ref)
        self._draw_force_and_moment(tot_thrust, tot_angle, tot_angular_thrust)
        self._draw_velocities(u_hat, v_hat, r_hat)
        self.draw_compass(heading_deg=psi_tilde - target_pose[2])
        self.draw_legend(self.legend_surface, self.box_pos)

        # final render
        self.render_surface_title()
        self.render_window()
