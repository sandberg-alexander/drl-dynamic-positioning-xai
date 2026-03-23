"""Smoke tests for the rendering package decomposition."""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"  # Must be before pygame import

import numpy as np
import pygame
import pytest


@pytest.fixture(scope="module", autouse=True)
def pygame_init():
    """Init pygame with dummy video driver for headless testing."""
    pygame.init()
    yield
    pygame.quit()


class TestRenderingImport:
    def test_render_explanation_importable(self):
        from milliampere_xai.rendering import RenderExplanation

        assert RenderExplanation is not None

    def test_backward_compat_import(self):
        with pytest.warns(DeprecationWarning):
            from milliampere_xai.dashboard import RenderExplanation as RE  # noqa: F811

        assert RE is not None


class TestRenderExplanationSmoke:
    def test_render_frame_zeros(self):
        from milliampere_xai.rendering import RenderExplanation

        render = RenderExplanation()

        render.render_frame(
            shap_values_action=[[0.0] * 14 for _ in range(8)],
            shap_values_value=[0.0] * 14,
            actuator_ref=[(0.0, 0.0)] * 4,
            tot_thrust=0.0,
            tot_angle=0.0,
            tot_angular_thrust=0.0,
            x_tilde=0.0,
            y_tilde=0.0,
            psi_tilde=0.0,
            u_hat=0.0,
            v_hat=0.0,
            r_hat=0.0,
            base_vectors=[0.0] * 8,
            action_low=[-1.0] * 8,
            action_high=[1.0] * 8,
            target_pose=(0.0, 0.0, 0.0),
            time_step=0,
            time=0,
        )

    def test_render_frame_realistic(self):
        from milliampere_xai.rendering import RenderExplanation

        render = RenderExplanation()

        render.render_frame(
            shap_values_action=[np.random.randn(14).tolist() for _ in range(8)],
            shap_values_value=np.random.randn(14).tolist(),
            actuator_ref=[(0.5, 45.0), (0.3, -30.0), (0.7, 120.0), (0.2, -90.0)],
            tot_thrust=0.6,
            tot_angle=45.0,
            tot_angular_thrust=0.3,
            x_tilde=1.5,
            y_tilde=-0.8,
            psi_tilde=15.0,
            u_hat=0.2,
            v_hat=-0.1,
            r_hat=5.0,
            base_vectors=np.random.randn(8).tolist(),
            action_low=[-1.0] * 8,
            action_high=[1.0] * 8,
            target_pose=(5.0, 3.0, 30.0),
            time_step=100,
            time=25,
            epsilon_ned=(1.2, -0.5, 10.0),
        )
