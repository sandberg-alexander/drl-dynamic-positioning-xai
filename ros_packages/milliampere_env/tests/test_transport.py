"""Tests for MockTransport."""

from __future__ import annotations

import numpy as np
import pytest

from milliampere_env.transport import MockTransport


class TestMockTransport:
    def test_initial_pose(self):
        t = MockTransport(initial_pose=(1.0, 2.0, 45.0))
        assert t.get_pose() == (1.0, 2.0, 45.0)

    def test_set_pose_roundtrip(self):
        t = MockTransport()
        t.set_pose(5.0, -3.0, 90.0)
        assert t.get_pose() == (5.0, -3.0, 90.0)

    def test_timestamp_increments(self):
        t = MockTransport(dt=0.1)
        t1 = t.get_timestamp()
        t2 = t.get_timestamp()
        assert t2 > t1
        assert t2 - t1 == pytest.approx(0.1)

    def test_reset_clears_pose(self):
        t = MockTransport(initial_pose=(5.0, 5.0, 90.0))
        t.reset_simulation(num_calls=1, sleep_between=0.0)
        assert t.get_pose() == (0.0, 0.0, 0.0)

    def test_reset_clears_time(self):
        t = MockTransport()
        t.get_timestamp()  # advance time
        t.reset_simulation(num_calls=1, sleep_between=0.0)
        assert t.get_timestamp() == pytest.approx(0.1)  # first tick after reset

    def test_sleep_is_noop(self):
        t = MockTransport()
        t.sleep(100.0)  # should not block

    def test_not_shutdown_by_default(self):
        t = MockTransport()
        assert t.is_shutdown() is False

    def test_publish_stores_values(self):
        t = MockTransport()
        throttle = np.array([100, 200, 300, 400], dtype=float)
        angles = np.array([135, -135, -45, 45], dtype=float)
        t.publish_actuator_setpoints(throttle, angles)
        np.testing.assert_array_equal(t._last_throttle, throttle)
        np.testing.assert_array_equal(t._last_angles, angles)

    def test_waypoint_default_none(self):
        t = MockTransport()
        assert t.get_waypoint() is None

    def test_set_waypoint(self):
        t = MockTransport()
        t.set_waypoint(10.0, 20.0, 45.0)
        assert t.get_waypoint() == (10.0, 20.0, 45.0)

    def test_mode_default_none(self):
        t = MockTransport()
        assert t.get_mode() is None

    def test_set_mode(self):
        t = MockTransport()
        t.set_mode("drl")
        assert t.get_mode() == "drl"

    def test_velocity_default_none(self):
        t = MockTransport()
        assert t.get_velocity() is None

    def test_set_velocity(self):
        t = MockTransport()
        t.set_velocity(0.5, -0.1, 0.02)
        assert t.get_velocity() == (0.5, -0.1, 0.02)

    def test_velocity_used_over_pose_delta(self):
        """When velocity is set, get_velocity returns it (not None)."""
        t = MockTransport()
        t.set_velocity(1.0, 0.0, 0.0)
        vel = t.get_velocity()
        assert vel is not None
        assert vel[0] == 1.0
