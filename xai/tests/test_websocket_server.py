"""Tests for the WebSocket server and frame broadcasting."""

from __future__ import annotations

import msgpack
import pytest
from fastapi.testclient import TestClient
from milliampere_dp.rendering.frames import (
    ActuatorState,
    KeyboardInput,
    RenderFrame,
    ShapFrame,
    VesselState,
)

from milliampere_xai.server.app import create_app
from milliampere_xai.server.connection_manager import ConnectionManager

# --- Fixtures ---


def _make_render_frame() -> RenderFrame:
    """Create a minimal valid RenderFrame for testing."""
    return RenderFrame(
        vessel=VesselState(
            x_tilde=1.0,
            y_tilde=2.0,
            psi_tilde=3.0,
            u_hat=0.1,
            v_hat=0.02,
            r_hat=0.5,
            target_pose=(10.0, 20.0, 45.0),
            epsilon_ned=(0.5, -0.3, 1.2),
        ),
        actuators=ActuatorState(
            actuator_ref=[(0.8, 30.0), (0.6, -20.0), (0.7, 10.0), (0.5, -15.0)],
            tot_thrust=2.5,
            tot_angle=12.0,
            tot_angular_thrust=0.8,
        ),
        shap=ShapFrame(
            shap_values_action=[[0.1] * 14 for _ in range(8)],
            shap_values_value=[0.05] * 14,
            base_vectors=[0.0] * 8,
            action_low=[-1.0] * 8,
            action_high=[1.0] * 8,
        ),
        time_step=42,
        time_seconds=10.5,
    )


@pytest.fixture
def test_frame() -> RenderFrame:
    return _make_render_frame()


@pytest.fixture
def app_and_manager(tmp_path):
    """Create a test app with a temporary dist directory."""
    # Create minimal dist dir with index.html
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>test</body></html>")
    keyboard_events: list[int] = []
    app, manager = create_app(
        on_keyboard=lambda key: keyboard_events.append(key),
        dist_dir=dist,
    )
    return app, manager, keyboard_events


# --- ConnectionManager unit tests ---


class TestConnectionManager:
    def test_initial_state(self):
        mgr = ConnectionManager()
        assert mgr.client_count == 0

    def test_keyboard_callback(self):
        events: list[int] = []
        mgr = ConnectionManager(on_keyboard=lambda k: events.append(k))
        mgr.handle_keyboard(3)
        assert events == [3]

    def test_keyboard_callback_none(self):
        mgr = ConnectionManager(on_keyboard=None)
        mgr.handle_keyboard(0)  # Should not raise


# --- Frame serialization round-trip ---


class TestFrameSerialization:
    def test_render_frame_msgpack_roundtrip(self, test_frame: RenderFrame):
        """Pydantic -> msgpack -> dict -> Pydantic round-trip."""
        packed = msgpack.packb(test_frame.model_dump())
        unpacked = msgpack.unpackb(packed)
        reconstructed = RenderFrame(**unpacked)
        assert reconstructed.time_step == test_frame.time_step
        assert reconstructed.vessel.x_tilde == test_frame.vessel.x_tilde
        assert len(reconstructed.shap.shap_values_action) == 8
        assert len(reconstructed.shap.shap_values_value) == 14

    def test_keyboard_input_msgpack_roundtrip(self):
        kb = KeyboardInput(key=3)
        packed = msgpack.packb(kb.model_dump())
        unpacked = msgpack.unpackb(packed)
        reconstructed = KeyboardInput(**unpacked)
        assert reconstructed.key == 3

    def test_render_frame_msgpack_is_bytes(self, test_frame: RenderFrame):
        """Verify msgpack produces bytes suitable for WebSocket binary."""
        packed = msgpack.packb(test_frame.model_dump())
        assert isinstance(packed, bytes)
        assert len(packed) > 0


# --- WebSocket endpoint integration tests ---


class TestWebSocketEndpoint:
    def test_static_index(self, app_and_manager):
        app, _manager, _events = app_and_manager
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "test" in resp.text

    def test_websocket_connect_disconnect(self, app_and_manager):
        app, manager, _events = app_and_manager
        client = TestClient(app)
        with client.websocket_connect("/ws") as _ws:
            assert manager.client_count == 1
        # After disconnect, count should be 0
        assert manager.client_count == 0

    def test_websocket_receive_keyboard(self, app_and_manager):
        app, _manager, events = app_and_manager
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            kb = KeyboardInput(key=2)
            ws.send_bytes(msgpack.packb(kb.model_dump()))
            # Give the server a moment to process
            import time

            time.sleep(0.05)
        assert 2 in events

    def test_websocket_invalid_keyboard(self, app_and_manager):
        """Invalid keyboard input should not crash the server."""
        app, _manager, events = app_and_manager
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(msgpack.packb({"key": 99}))
            import time

            time.sleep(0.05)
        assert len(events) == 0  # Invalid key (>5) rejected
