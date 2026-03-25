"""SHAP value visualization with per-thruster force/moment decomposition.

Extracted from dashboard.py ShapExplainRender. Dead rendering modes
(_draw_explanation, _draw_explanation2, _draw_explanation3) and their
helper methods removed — only the active _draw_explanation4 path is kept.
"""

from __future__ import annotations

import numpy as np
import pygame
from milliampere_dp.rendering import Color
from milliampere_dp.transforms import ssa
from milliampere_dp.vessel import THRUSTER_ARM_X, THRUSTER_ARM_Y, VESSEL_LENGTH

from milliampere_xai.model_wrappers import combine_actuator_ref
from milliampere_xai.rendering._primitives import Utilities
from milliampere_xai.rendering._vessel import VesselRender


class ShapExplainRender(VesselRender, Utilities):
    """SHAP value visualization with per-thruster force/moment decomposition."""

    SCALE = 50
    ACTUATOR_X = THRUSTER_ARM_X  # from milliampere_dp.vessel (1.8 m)
    ACTUATOR_Y = THRUSTER_ARM_Y  # from milliampere_dp.vessel (0.8 m)
    VESSEL_MOMENT_MARKER = 50 / SCALE

    def __init__(
        self,
        screen,
        window_pos,
        title="Desired total trust force/moment explained in BODY-frame",
        window_width=450,
        window_height=450,
        renderer=None,
    ):
        VesselRender.__init__(
            self,
            screen,
            window_pos,
            self.SCALE,
            title,
            window_width,
            window_height,
            renderer=renderer,
        )
        Utilities.__init__(self)

        self.label_font = pygame.font.SysFont("DejaVu Sans", 12)

        self.body_legend_items = (
            (
                Color.RED.value,
                "Main feature causing the desired total trust force/moment",
            ),
            (
                Color.DESIRED_LIGHT_YELLOW.value,
                "Estimated desired total trust force/moment based on main feature",
            ),
        )

        self.legend_surface, self.box_pos = self.create_legend(
            self.body_legend_items,
            self.label_font,
            marker_size=12,
            spacing=5,
            padding=5,
            margin=5,
            padding_bottom=0,
        )

        self.idx = 0
        self.prev_idx = -1
        self.explain_offset = 20
        self.prev_angle_1 = np.array([135, -135, -45, 45])
        self.prev_angle_2 = np.array([135, -135, -45, 45])

        self.actuator_pos = np.array(
            [
                [self.ACTUATOR_X, -self.ACTUATOR_Y],
                [self.ACTUATOR_X, self.ACTUATOR_Y],
                [-self.ACTUATOR_X, self.ACTUATOR_Y],
                [-self.ACTUATOR_X, -self.ACTUATOR_Y],
            ]
        )

    def _find_explanation2(
        self, shap_values_action, base_value, action_low, action_high
    ):
        sv_action = np.array(shap_values_action)
        sv_base = np.array(base_value)
        act_low = np.array(action_low)
        act_high = np.array(action_high)

        sv_all_actions = np.clip(sv_action.sum(axis=1) + sv_base, act_low, act_high)

        pos_totalt = np.where(sv_action > 0, sv_action, 0).sum(axis=1) + np.where(
            sv_base > 0, sv_base, 0
        )
        neg_totalt = np.where(sv_action < 0, sv_action, 0).sum(axis=1) + np.where(
            sv_base < 0, sv_base, 0
        )

        denominators = np.where(act_high == 1, pos_totalt, neg_totalt)

        c = np.zeros_like(sv_all_actions, dtype=float)

        np.divide(
            sv_all_actions, denominators, out=c, where=denominators != 0
        )  # Avoid division by zero

        num_actions, num_feats = sv_action.shape
        assert num_actions % 2 == 0, "Need an even number of action-rows"
        sv_pairs = sv_action.reshape(-1, 2, num_feats)  # shape (4, 2, 14)
        c_pairs = c.reshape(-1, 2)  # shape (4, 2)

        # Scale each x-row by its c and each y-row by its c
        scaled = sv_pairs * c_pairs[:, :, None]  # shape (4, 2, 14)
        self.scaled = scaled

        base_scaled = (sv_base[None, None, :] * c_pairs[:, :, None]).sum(
            axis=2, keepdims=True
        )
        self.scaled_with_base = np.concatenate([scaled, base_scaled], axis=2)

        dists = np.hypot(scaled[:, 0, :], scaled[:, 1, :])

        # Clip each distance between 0 and 1
        dists_clipped = np.clip(dists, 0, 1)  # shape (4, 14)
        dists_tot_clipped = np.clip(dists, 0, 1)
        sumed_features = dists_clipped.sum(axis=0)
        max_idx = np.argmax(sumed_features)
        self.idx = max_idx
        self.dists_clipped = dists_clipped
        self.dists_tot_clipped = dists_tot_clipped

    def _draw_explanation4(
        self,
        shap_values_action,
        shap_values_value,
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
        base_vectors,
        action_low,
        action_high,
    ):
        thrusters_shap = self.dists_clipped[:, self.idx]

        angles_shap = np.where(
            thrusters_shap == 0,
            self.prev_angle_1,
            np.rad2deg(
                np.arctan2(self.scaled[:, 1, self.idx], self.scaled[:, 0, self.idx])
            ),
        )
        if angles_shap[1] == 180:
            angles_shap[1] = -180

        self.prev_angle_1 = angles_shap.copy()
        vec = list(zip(thrusters_shap, angles_shap))

        thrusters_shap_tot = self.dists_tot_clipped.sum(axis=1)
        angles_shap2 = np.where(
            thrusters_shap_tot == 0,
            self.prev_angle_2,
            np.rad2deg(
                np.arctan2(
                    self.scaled_with_base.sum(axis=2)[:, 1],
                    self.scaled_with_base.sum(axis=2)[:, 0],
                )
            ),
        )
        if angles_shap2[1] == 180:
            angles_shap2[1] = -180

        self.prev_angle_2 = angles_shap2.copy()
        vec2 = list(zip(thrusters_shap_tot, angles_shap2))

        vector = combine_actuator_ref(vec, self.actuator_pos)
        combine_actuator_ref(vec2, self.actuator_pos)

        radius = 30
        raw_angle = vector[2] / radius
        raw_angle = np.clip(raw_angle, -2 * np.pi, 2 * np.pi)
        draw_angle = self._scalar2pygame(raw_angle)
        self._draw_curved_arrow(
            draw_angle, self.center, Color.DESIRED_LIGHT_YELLOW.value, radius=radius
        )

        self._draw_arrow(
            self._degrees2pygame(vector[1]),
            self._scalar2pygame(vector[0]),
            Color.DESIRED_LIGHT_YELLOW.value,
            self.center,
        )

        if self.idx == 0:
            self._draw_distance(
                self.surface,
                (
                    self.center[0]
                    + self._scalar2pygame(self.VESSEL_WIDTH / 2)
                    + self.explain_offset,
                    self.center[1],
                ),
                self._scalar2pygame(x_tilde),
                self._degrees2pygame(0),
                Color.RED.value,
                label=f"{x_tilde:.2f} m",
            )
        elif self.idx == 1:
            self._draw_distance(
                self.surface,
                (
                    self.center[0],
                    self.center[1]
                    + self._scalar2pygame(VESSEL_LENGTH / 2)
                    + self.explain_offset,
                ),
                self._scalar2pygame(y_tilde),
                self._degrees2pygame(90),
                Color.RED.value,
                label=f"{y_tilde:.2f} m",
            )
        elif self.idx == 2:
            self._draw_angle(
                self.surface,
                self.center,
                np.deg2rad(psi_tilde),
                Color.RED.value,
                radius=self._scalar2pygame(VESSEL_LENGTH / 2) + self.explain_offset,
                label=f"{psi_tilde:.0f} \u00b0",
            )
        elif self.idx == 3:
            if u_hat > 0:
                self._draw_arrow(
                    self._degrees2pygame(0),
                    self._scalar2pygame(u_hat),
                    Color.RED.value,
                    (
                        self.center[0],
                        self.center[1]
                        - self._scalar2pygame(VESSEL_LENGTH / 2)
                        - self.explain_offset,
                    ),
                    show_measurement=True,
                    label=f"{u_hat / 2:.2f} m/s",
                )
            else:
                self._draw_arrow(
                    self._degrees2pygame(0),
                    self._scalar2pygame(u_hat),
                    Color.RED.value,
                    (
                        self.center[0],
                        self.center[1]
                        + self._scalar2pygame(VESSEL_LENGTH / 2)
                        + self.explain_offset,
                    ),
                    show_measurement=True,
                    label=f"{u_hat / 2:.2f} m/s",
                )
        elif self.idx == 4:
            if v_hat > 0:
                self._draw_arrow(
                    self._degrees2pygame(90),
                    self._scalar2pygame(v_hat),
                    Color.RED.value,
                    (
                        self.center[0]
                        + self._scalar2pygame(self.VESSEL_WIDTH / 2)
                        + self.explain_offset,
                        self.center[1],
                    ),
                    show_measurement=True,
                    label=f"{v_hat / 2:.2f} m/s",
                )
            else:
                self._draw_arrow(
                    self._degrees2pygame(90),
                    self._scalar2pygame(v_hat),
                    Color.RED.value,
                    (
                        self.center[0]
                        - self._scalar2pygame(self.VESSEL_WIDTH / 2)
                        - self.explain_offset,
                        self.center[1],
                    ),
                    show_measurement=True,
                    label=f"{v_hat / 2:.2f} m/s",
                )
        elif self.idx == 5:
            self._draw_curved_arrow(
                np.deg2rad(r_hat),
                self.center,
                Color.RED.value,
                radius=self._scalar2pygame(VESSEL_LENGTH / 2) + self.explain_offset,
                show_measurement=True,
                label=f"{r_hat * 112.6 / (360 * 2):.0f} \u00b0/s",
            )

        error = np.array([tot_thrust, tot_angle, tot_angular_thrust]) - np.array(
            [vector[0], vector[1], vector[2]]
        )

        acc_RPM = ((np.sqrt(2) + 1) - abs(error[0])) / (np.sqrt(2) + 1)
        acc_angle = (180 - abs(ssa(tot_angle - vector[1]))) / 180
        acc_moment = (10.4 - abs(error[2])) / (10.4)

        acc_total = (acc_RPM + acc_angle + acc_moment) / 3

        if acc_RPM > 0.75:
            acc_RPM_color = Color.GREEN.value
        elif acc_RPM > 0.5:
            acc_RPM_color = Color.ORANGE.value
        else:
            acc_RPM_color = Color.RED.value

        if acc_angle > 0.75:
            acc_angle_color = Color.GREEN.value
        elif acc_angle > 0.5:
            acc_angle_color = Color.ORANGE.value
        else:
            acc_angle_color = Color.RED.value

        if acc_moment > 0.75:
            acc_moment_color = Color.GREEN.value
        elif acc_moment > 0.5:
            acc_moment_color = Color.ORANGE.value
        else:
            acc_moment_color = Color.RED.value

        if acc_total > 0.75:
            acc_total_color = Color.GREEN.value
        elif acc_total > 0.5:
            acc_total_color = Color.ORANGE.value
        else:
            acc_total_color = Color.RED.value

        text_surface1 = self.window_font.render(
            f"Explained RPM: {acc_RPM:.2f}", self.antialias, acc_RPM_color
        )
        text_rect1 = text_surface1.get_rect()
        text_rect1.topleft = (self._title_offset, self._title_offset * 10)

        text_surface2 = self.window_font.render(
            f"Explained angle: {acc_angle:.2f}", self.antialias, acc_angle_color
        )
        text_rect2 = text_surface2.get_rect()
        text_rect2.topleft = (
            self._title_offset,
            self._title_offset * 11 + text_rect1.height,
        )

        text_surface3 = self.window_font.render(
            f"Explained moment: {acc_moment:.2f}", self.antialias, acc_moment_color
        )
        text_rect3 = text_surface3.get_rect()
        text_rect3.topleft = (
            self._title_offset,
            self._title_offset * 12 + text_rect1.height + text_rect2.height,
        )

        text_surface4 = self.window_font.render(
            f"Explained TOTAL: {acc_total:.2f}", self.antialias, acc_total_color
        )
        text_rect4 = text_surface4.get_rect()
        text_rect4.topleft = (
            self._title_offset,
            self._title_offset * 14
            + text_rect1.height
            + text_rect2.height
            + text_rect3.height,
        )

        self.surface.blit(text_surface1, text_rect1)
        self.surface.blit(text_surface2, text_rect2)
        self.surface.blit(text_surface3, text_rect3)
        self.surface.blit(text_surface4, text_rect4)

    def render(
        self,
        shap_values_action,
        shap_values_value,
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
        base_vectors,
        action_low,
        action_high,
    ):
        self._find_explanation2(
            shap_values_action, base_vectors, action_low, action_high
        )

        self.render_surface()

        self.draw_target(x_tilde, y_tilde, psi_tilde, color=Color.GRAY.value)
        self.draw_vessel()
        self._draw_explanation4(
            shap_values_action,
            shap_values_value,
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
            base_vectors,
            action_low,
            action_high,
        )
        self.draw_legend(self.legend_surface, self.box_pos)

        self.render_surface_title()
        self.render_window()
