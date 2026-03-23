"""Pydantic schemas for rendering frame data.

Typed contract between the environment/XAI pipeline and the rendering
layer.  Used by the future web backend (Phase 3) for WebSocket payloads.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VesselState(BaseModel):
    """Vessel position, heading, and velocity state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x_tilde: float = Field(description="Surge error in body frame [m]")
    y_tilde: float = Field(description="Sway error in body frame [m]")
    psi_tilde: float = Field(description="Heading error [deg]")
    u_hat: float = Field(default=0.0, description="Surge velocity [m/s]")
    v_hat: float = Field(default=0.0, description="Sway velocity [m/s]")
    r_hat: float = Field(default=0.0, description="Yaw rate [deg/s]")
    target_pose: tuple[float, float, float] = Field(
        description="Target (north_m, east_m, heading_deg)"
    )
    epsilon_ned: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="NED-frame error (north_m, east_m, heading_deg)",
    )


class ActuatorState(BaseModel):
    """Thruster commands and aggregate force/moment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actuator_ref: list[tuple[float, float]] = Field(
        description="Per-thruster (thrust_normalised, angle_deg)"
    )
    tot_thrust: float = Field(description="Total thrust magnitude")
    tot_angle: float = Field(description="Total thrust angle [deg]")
    tot_angular_thrust: float = Field(description="Total angular moment")


class ShapFrame(BaseModel):
    """SHAP explanation data for a single frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shap_values_action: list[list[float]] = Field(
        description="Action SHAP values (8 outputs x 14 features)"
    )
    shap_values_value: list[float] = Field(
        description="Value function SHAP values (14 features)"
    )
    base_vectors: list[float] = Field(description="Expected action values (8 outputs)")
    action_low: list[float] = Field(description="Action lower bounds (8)")
    action_high: list[float] = Field(description="Action upper bounds (8)")


class RenderFrame(BaseModel):
    """Complete frame payload for rendering."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vessel: VesselState
    actuators: ActuatorState
    shap: ShapFrame
    time_step: int = Field(ge=0)
    time_seconds: float = Field(ge=0.0)


class KeyboardInput(BaseModel):
    """Client-to-server keyboard event (deploy mode switching)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: int = Field(ge=0, le=5, description="Deploy mode (0-5)")
