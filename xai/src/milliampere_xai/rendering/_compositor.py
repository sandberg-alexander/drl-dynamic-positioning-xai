"""Top-level compositor that creates and manages all XAI dashboard sub-windows."""

from __future__ import annotations

import pygame
from milliampere_dp.rendering import Color
from milliampere_dp.rendering.pygame_renderer import PygameRenderer

from milliampere_xai.rendering._body import BodyRender
from milliampere_xai.rendering._ned import NedRender
from milliampere_xai.rendering._shap_bars import ShapRender
from milliampere_xai.rendering._shap_explain import ShapExplainRender


class RenderExplanation:
    """Top-level compositor that creates and manages all XAI dashboard sub-windows.

    Instantiates BodyRender, NedRender, ShapExplainRender, and ShapRender windows,
    then composites them into a single pygame display each frame via render_frame().
    """

    SCREEN_WIDTH = 975
    SCREEN_HEIGHT = 975
    TITLE = "Explanations"

    def __init__(self):
        self._renderer = PygameRenderer()
        self._screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption(self.TITLE)

        self.shap_legend_items1 = (
            "n_{\u2081,d,\u209c} [RPM]",
            "n_{\u2082,d,\u209c} [RPM]",
            "n_{\u2083,d,\u209c} [RPM]",
            "n_{\u2084,d,\u209c} [RPM]",
        )

        self.shap_legend_items2 = (
            "\u03b1_{d\u2081,\u209c} [\u00b0]",
            "\u03b1_{d\u2082,\u209c} [\u00b0]",
            "\u03b1_{d\u2083,\u209c} [\u00b0]",
            "\u03b1_{d\u2084,\u209c} [\u00b0]",
        )

        self._ned_window = NedRender(
            self._screen, window_pos=(25, 25), renderer=self._renderer
        )
        self._body_window = BodyRender(
            self._screen, window_pos=(25, 500), renderer=self._renderer
        )
        self._shap_window_top = ShapRender(
            self._screen,
            window_pos=(500, 25),
            legend_items=self.shap_legend_items1,
            title="Feature importance using thrust SHAP-values",
            renderer=self._renderer,
        )
        self._shap_window_bottom = ShapRender(
            self._screen,
            window_pos=(500, 500),
            legend_items=self.shap_legend_items2,
            title="SHAP-values azimuth angles",
            window_width=475,
            renderer=self._renderer,
        )

        self._shap_explain_window = ShapExplainRender(
            self._screen, window_pos=(500, 500), renderer=self._renderer
        )

        self.window_font = pygame.font.SysFont("DejaVu Sans", 11)

    def render_frame(
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
        target_pose,
        time_step,
        time,
        epsilon_ned=None,
    ):
        # print(sum(shap_values_value))
        if sum(shap_values_value) < -2:
            self._screen.fill(Color.RED.value)
        else:
            self._screen.fill(Color.SCREEN_COLOR.value)

        text_surface = self.window_font.render(
            f"time step: {int(time_step)}, time: {time} s", True, Color.BLACK.value
        )
        text_rect = text_surface.get_rect()
        text_rect.center = (int(self.SCREEN_WIDTH / 2), self.SCREEN_HEIGHT - 15)
        self._screen.blit(text_surface, text_rect)

        self._body_window.render(
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
        )
        # NED window uses actual NED-frame errors (not capped body-frame)
        if epsilon_ned is not None:
            self._ned_window.render(
                epsilon_ned[0], epsilon_ned[1], epsilon_ned[2], target_pose
            )
        else:
            self._ned_window.render(x_tilde, y_tilde, psi_tilde, target_pose)
        self._shap_explain_window.render(
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

        self._shap_window_top.render(
            shap_values_action, shap_values_value, base_vectors, action_low, action_high
        )
        pygame.display.flip()
