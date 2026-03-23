"""Tests for milliampere_dp.rendering.frames."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from milliampere_dp.rendering.frames import (  # noqa: E402
    ActuatorState,
    KeyboardInput,
    RenderFrame,
    ShapFrame,
    VesselState,
)


def _make_vessel_state(**overrides):
    defaults = dict(
        x_tilde=0.5,
        y_tilde=-0.3,
        psi_tilde=10.0,
        u_hat=0.1,
        v_hat=0.0,
        r_hat=0.5,
        target_pose=(100.0, 50.0, 45.0),
        epsilon_ned=(0.4, -0.2, 10.0),
    )
    defaults.update(overrides)
    return VesselState(**defaults)


def _make_actuator_state():
    return ActuatorState(
        actuator_ref=[(0.5, 30.0), (0.3, -45.0), (0.7, 60.0), (0.4, -30.0)],
        tot_thrust=1.2,
        tot_angle=15.0,
        tot_angular_thrust=0.3,
    )


def _make_shap_frame():
    return ShapFrame(
        shap_values_action=[[0.1] * 14 for _ in range(8)],
        shap_values_value=[0.05] * 14,
        base_vectors=[0.5] * 8,
        action_low=[0.0] * 8,
        action_high=[1.0] * 8,
    )


class TestVesselState:
    def test_roundtrip(self):
        vs = _make_vessel_state()
        d = vs.model_dump()
        vs2 = VesselState(**d)
        assert vs == vs2

    def test_frozen(self):
        vs = _make_vessel_state()
        with pytest.raises(pydantic.ValidationError):
            vs.x_tilde = 99.0  # type: ignore[misc]

    def test_extra_forbidden(self):
        with pytest.raises(pydantic.ValidationError):
            _make_vessel_state(bogus_field=42)


class TestActuatorState:
    def test_roundtrip(self):
        a = _make_actuator_state()
        assert ActuatorState(**a.model_dump()) == a


class TestShapFrame:
    def test_roundtrip(self):
        s = _make_shap_frame()
        assert ShapFrame(**s.model_dump()) == s


class TestRenderFrame:
    def test_roundtrip(self):
        frame = RenderFrame(
            vessel=_make_vessel_state(),
            actuators=_make_actuator_state(),
            shap=_make_shap_frame(),
            time_step=42,
            time_seconds=4.2,
        )
        d = frame.model_dump()
        assert RenderFrame(**d) == frame

    def test_frozen(self):
        frame = RenderFrame(
            vessel=_make_vessel_state(),
            actuators=_make_actuator_state(),
            shap=_make_shap_frame(),
            time_step=0,
            time_seconds=0.0,
        )
        with pytest.raises(pydantic.ValidationError):
            frame.time_step = 99  # type: ignore[misc]

    def test_negative_timestep_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            RenderFrame(
                vessel=_make_vessel_state(),
                actuators=_make_actuator_state(),
                shap=_make_shap_frame(),
                time_step=-1,
                time_seconds=0.0,
            )


class TestKeyboardInput:
    @pytest.mark.parametrize("key", [0, 1, 2, 3, 4, 5])
    def test_valid_keys(self, key):
        ki = KeyboardInput(key=key)
        assert ki.key == key

    def test_out_of_range(self):
        with pytest.raises(pydantic.ValidationError):
            KeyboardInput(key=6)

    def test_negative(self):
        with pytest.raises(pydantic.ValidationError):
            KeyboardInput(key=-1)
