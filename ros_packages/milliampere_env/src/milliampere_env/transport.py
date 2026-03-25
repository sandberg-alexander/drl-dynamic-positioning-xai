"""Transport abstraction for vessel communication.

VesselTransport defines the interface. RosTransport talks to the real
ROS stack. MockTransport enables testing without ROS, Docker, or a
simulator.

RosTransport lazy-imports rospy so that milliampere_env can be imported
and tested on the host (with MockTransport) without ROS installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from milliampere_env.config import EnvConfig


class VesselTransport(ABC):
    """Abstract interface between the gym env and the vessel/simulator."""

    @abstractmethod
    def publish_actuator_setpoints(
        self, throttle_rpm: np.ndarray, angles_deg: np.ndarray
    ) -> None:
        """Publish thruster commands (4 throttle RPM values, 4 angle degrees)."""

    @abstractmethod
    def get_pose(self) -> tuple[float, float, float]:
        """Return current vessel pose (x_north, y_east, psi_deg)."""

    @abstractmethod
    def get_timestamp(self) -> float:
        """Return current observation timestamp in seconds."""

    def get_velocity(self) -> tuple[float, float, float] | None:
        """Return body-frame velocity (u, v, r) or None for pose-delta estimation."""
        return None

    @abstractmethod
    def reset_simulation(self, num_calls: int, sleep_between: float) -> None:
        """Call the simulator reset service."""

    @abstractmethod
    def sleep(self, duration: float) -> None:
        """Sleep for the given duration (seconds)."""

    @abstractmethod
    def is_shutdown(self) -> bool:
        """Return True if the transport/ROS is shutting down."""

    def get_waypoint(self) -> tuple[float, float, float] | None:
        """Return (north, east, heading_deg) from guidance, or None."""
        return None

    def get_mode(self) -> str | None:
        """Return current supervisor mode string, or None."""
        return None

    def wait_for_pose(self, timeout: float = 30.0) -> None:
        """Block until the first pose message is received."""
        pass

    def publish_env_state(self, state_dict: dict) -> None:
        """Publish env state for the standalone viewer. No-op by default."""
        pass


class RosTransport(VesselTransport):
    """Real ROS transport. Lazy-imports rospy to avoid host-side import errors."""

    def __init__(self, config: EnvConfig) -> None:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from scipy.spatial.transform import Rotation

        self._rospy = rospy
        self._Rotation = Rotation
        self._config = config

        # State from callbacks
        self._pose_data = None
        self._twist_data = None
        self._waypoint_data = None
        self._mode_data = None

        # Publishers (4 thrusters)
        from custom_msgs.msg import ActuatorSetpoints

        self._ActuatorSetpoints = ActuatorSetpoints
        self._publishers = [
            rospy.Publisher(f"/actuator_ref_{i + 1}", ActuatorSetpoints, queue_size=1)
            for i in range(4)
        ]

        # Subscribers
        rospy.Subscriber("/navigation/pose", PoseStamped, self._pose_callback)

        if config.velocity.use_topic:
            from geometry_msgs.msg import TwistStamped

            rospy.Subscriber(config.velocity.topic, TwistStamped, self._twist_callback)

        if config.target.type.value == "waypoint":
            from custom_msgs.msg import NorthEastHeading

            rospy.Subscriber(
                "/guidance/waypoint", NorthEastHeading, self._waypoint_callback
            )

        if config.mode_checking.enabled:
            from std_msgs.msg import String

            rospy.Subscriber(config.mode_checking.topic, String, self._mode_callback)

        # State publisher for viewer
        try:
            from custom_ros_msgs.msg import EnvState  # type: ignore[attr-defined]

            self._EnvState = EnvState
            self._state_pub = rospy.Publisher(
                "/milliampere_env/state", EnvState, queue_size=1
            )
        except ImportError:
            self._EnvState = None
            self._state_pub = None

        # Reset service proxy (created lazily on first call)
        self._reset_proxy = None

    def _pose_callback(self, data) -> None:
        self._pose_data = data

    def _twist_callback(self, data) -> None:
        self._twist_data = data

    def _waypoint_callback(self, data) -> None:
        self._waypoint_data = data

    def _mode_callback(self, data) -> None:
        self._mode_data = data

    def publish_actuator_setpoints(
        self, throttle_rpm: np.ndarray, angles_deg: np.ndarray
    ) -> None:
        # Gate on mode if configured
        if self._config.mode_checking.enabled:
            mode = self.get_mode()
            if mode != self._config.mode_checking.required_mode:
                return

        for i in range(4):
            msg = self._ActuatorSetpoints()
            msg.throttle_reference = round(throttle_rpm[i])
            msg.angle_reference = round(angles_deg[i])
            self._publishers[i].publish(msg)

    def get_pose(self) -> tuple[float, float, float]:
        if self._pose_data is None:
            raise RuntimeError("No pose data received yet")
        x = self._pose_data.pose.position.x
        y = self._pose_data.pose.position.y
        q = self._pose_data.pose.orientation
        rotation = self._Rotation.from_quat([q.x, q.y, q.z, q.w])
        psi_deg = float(rotation.as_euler("zxy", degrees=True)[0])
        return (x, y, psi_deg)

    def get_timestamp(self) -> float:
        if self._pose_data is None:
            raise RuntimeError("No pose data received yet")
        return self._pose_data.header.stamp.to_sec()

    def get_velocity(self) -> tuple[float, float, float] | None:
        if self._twist_data is None:
            return None
        t = self._twist_data.twist
        return (t.linear.x, t.linear.y, t.angular.z)

    def reset_simulation(self, num_calls: int, sleep_between: float) -> None:
        if self._reset_proxy is None:
            from sim_milliampere.srv import ResetState

            self._rospy.wait_for_service("/sim_vessel/reset_state")
            self._reset_proxy = self._rospy.ServiceProxy(
                "/sim_vessel/reset_state", ResetState
            )

        for i in range(num_calls):
            try:
                self._reset_proxy(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            except Exception as e:
                self._rospy.logwarn(f"Reset service call {i + 1} failed: {e}")
            if i < num_calls - 1:
                self.sleep(sleep_between)

    def sleep(self, duration: float) -> None:
        self._rospy.sleep(duration)

    def is_shutdown(self) -> bool:
        return self._rospy.is_shutdown()

    def get_waypoint(self) -> tuple[float, float, float] | None:
        if self._waypoint_data is None:
            return None
        return (
            self._waypoint_data.north,
            self._waypoint_data.east,
            float(np.rad2deg(self._waypoint_data.heading)),
        )

    def get_mode(self) -> str | None:
        if self._mode_data is None:
            return None
        return self._mode_data.data

    def wait_for_pose(self, timeout: float = 30.0) -> None:
        rate = self._rospy.Rate(10)
        elapsed = 0.0
        while self._pose_data is None and not self.is_shutdown():
            rate.sleep()
            elapsed += 0.1
            if elapsed >= timeout:
                raise TimeoutError("Timed out waiting for pose data")

    def publish_env_state(self, state_dict: dict) -> None:
        if self._state_pub is None or self._EnvState is None:
            return
        msg = self._EnvState()
        msg.epsilon = state_dict["epsilon"]
        msg.epsilon_ned = state_dict["epsilon_ned"]
        msg.est_velocity = state_dict["est_velocity"]
        msg.target_pose = state_dict["target_pose"]
        msg.thrusters = state_dict["thrusters"]
        msg.angles = state_dict["angles"]
        msg.reward_components = state_dict["reward_components"]
        msg.reward_total = state_dict["reward_total"]
        msg.time_step = state_dict["time_step"]
        msg.episode = state_dict["episode"]
        self._state_pub.publish(msg)


class MockTransport(VesselTransport):
    """Deterministic transport for testing. No ROS dependency."""

    def __init__(
        self,
        initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 0.1,
    ) -> None:
        self._pose = np.array(initial_pose, dtype=float)
        self._time = 0.0
        self._dt = dt
        self._shutdown = False
        self._last_throttle = np.zeros(4)
        self._last_angles = np.zeros(4)
        self._waypoint: tuple[float, float, float] | None = None
        self._velocity: tuple[float, float, float] | None = None
        self._mode: str | None = None
        self._reset_count = 0

    def publish_actuator_setpoints(
        self, throttle_rpm: np.ndarray, angles_deg: np.ndarray
    ) -> None:
        self._last_throttle = throttle_rpm.copy()
        self._last_angles = angles_deg.copy()

    def get_pose(self) -> tuple[float, float, float]:
        return tuple(self._pose)

    def get_timestamp(self) -> float:
        self._time += self._dt
        return self._time

    def reset_simulation(self, num_calls: int, sleep_between: float) -> None:
        self._pose[:] = 0.0
        self._time = 0.0
        self._reset_count += num_calls

    def sleep(self, duration: float) -> None:
        pass  # No-op for tests

    def is_shutdown(self) -> bool:
        return self._shutdown

    def get_velocity(self) -> tuple[float, float, float] | None:
        return self._velocity

    def get_waypoint(self) -> tuple[float, float, float] | None:
        return self._waypoint

    def get_mode(self) -> str | None:
        return self._mode

    def wait_for_pose(self, timeout: float = 30.0) -> None:
        pass  # Already has pose

    # --- Test helpers (not in ABC) ---

    def set_pose(self, x: float, y: float, psi_deg: float) -> None:
        """Inject a pose for testing."""
        self._pose[:] = [x, y, psi_deg]

    def set_velocity(self, u: float, v: float, r: float) -> None:
        """Inject a body-frame velocity for testing."""
        self._velocity = (u, v, r)

    def set_waypoint(self, north: float, east: float, heading_deg: float) -> None:
        """Inject a waypoint for testing."""
        self._waypoint = (north, east, heading_deg)

    def set_mode(self, mode: str) -> None:
        """Inject a supervisor mode for testing."""
        self._mode = mode
