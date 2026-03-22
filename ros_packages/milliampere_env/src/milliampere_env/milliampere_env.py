"""Consolidated MilliAmpere1 gymnasium environment.

Replaces 14 copy-pasted env files (v1-v13) with a single config-driven
class. All rendering is removed — use the standalone viewer instead.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from milliampere_dp import vessel
from milliampere_dp.log import get_logger
from milliampere_dp.rewards import (
    angle_rate_penalty,
    build_inv_sigma,
    gaussian_reward,
    thrust_penalty,
    thrust_rate_penalty,
    velocity_penalty,
)
from milliampere_dp.transforms import rotate_ned2body, ssa

from milliampere_env.config import EnvConfig, TargetType
from milliampere_env.transport import RosTransport, VesselTransport

logger = get_logger("env")


class MilliAmpereEnv(gym.Env):
    """Config-driven DP environment for milliAmpere1.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML config file. If None, uses default config.
    transport : VesselTransport, optional
        Inject a transport (e.g. MockTransport for testing).
        If None, creates a RosTransport from config.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path | None = None,
        transport: VesselTransport | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        # Load and validate config
        if config_path is not None:
            self.config = EnvConfig.from_yaml(config_path)
        else:
            self.config = EnvConfig()

        # Build action space (positive-only thrust)
        self.action_space = spaces.Box(
            low=vessel.LOW_ACTION.copy(),
            high=vessel.HIGH_ACTION.copy(),
            dtype=np.float64,
        )

        # Build observation space: 6 state dims + 8 prev-action dims = 14
        low_obs = np.concatenate([np.full(6, -1.0), vessel.LOW_ACTION])
        high_obs = np.concatenate([np.full(6, 1.0), vessel.HIGH_ACTION])
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float64
        )

        # Transport
        self.transport = transport or RosTransport(self.config)

        # Precompute inverse covariance matrices for reward
        rc = self.config.rewards
        self._inv_sigma = build_inv_sigma(rc.sigma_d, rc.sigma_psi)
        self._inv_sigma_AS = build_inv_sigma(rc.sigma_AS_d, rc.sigma_AS_psi)

        # Episode state (initialised in reset)
        self._time_step = 0
        self._episode = 0
        self._terminated_flag = False
        self._target_pose = np.zeros(3)

        # Vessel state
        self._eta = np.zeros(3)  # x, y, psi_deg (NED)
        self._eta_prev = np.zeros(3)
        self._epsilon_ned = np.zeros(3)  # error in NED
        self._epsilon = np.zeros(3)  # error in body frame
        self._est_nu = np.zeros(3)  # estimated velocity (u, v, r)
        self._obs_time: float | None = None
        self._obs_time_prev: float | None = None

        # Actuator state
        self._action = np.zeros(8)
        self._action_prev = np.zeros(8)
        self._norm_action = np.zeros(8)
        self._norm_action_prev = np.zeros(8)
        self._thrusters = np.zeros(4)
        self._thrusters_prev = np.zeros(4)
        self._angles = vessel.DEFAULT_INITIAL_ANGLES.copy()
        self._angles_prev = vessel.DEFAULT_INITIAL_ANGLES.copy()

        # Observation cache
        self._observation: np.ndarray | None = None
        self._norm_observation: np.ndarray | None = None

        # Reward component cache (for info dict / viewer)
        self._R_gauss = 0.0
        self._R_AS_gauss = 0.0
        self._R_vel = 0.0
        self._R_thrust = 0.0
        self._R_thrust_d = 0.0
        self._R_angle_d = 0.0

    # ------------------------------------------------------------------
    # Read-only state accessors
    # ------------------------------------------------------------------

    @property
    def target_pose(self) -> np.ndarray:
        """Current target pose [x_north, y_east, psi_deg] (read-only copy)."""
        return self._target_pose.copy()

    @property
    def thrusters(self) -> np.ndarray:
        """Current thruster RPM setpoints [n1, n2, n3, n4] (read-only copy)."""
        return self._thrusters.copy()

    @property
    def angles(self) -> np.ndarray:
        """Current thruster angles in degrees [a1, a2, a3, a4] (read-only copy)."""
        return self._angles.copy()

    @property
    def epsilon_ned(self) -> np.ndarray:
        """Position error in NED frame [dx, dy, dpsi_deg] (read-only copy)."""
        return self._epsilon_ned.copy()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._episode += 1
        self._time_step = 0
        self._terminated_flag = False

        # Reset state arrays
        self._eta[:] = 0.0
        self._eta_prev[:] = 0.0
        self._epsilon[:] = 0.0
        self._epsilon_ned[:] = 0.0
        self._est_nu[:] = 0.0
        self._obs_time = None
        self._obs_time_prev = None
        self._action[:] = 0.0
        self._action_prev[:] = 0.0
        self._norm_action[:] = 0.0
        self._norm_action_prev[:] = 0.0
        self._thrusters[:] = 0.0
        self._thrusters_prev[:] = 0.0
        self._angles[:] = vessel.DEFAULT_INITIAL_ANGLES
        self._angles_prev[:] = vessel.DEFAULT_INITIAL_ANGLES

        # Wait for mode if configured
        if self.config.mode_checking.enabled:
            while self.transport.get_mode() != self.config.mode_checking.required_mode:
                self.transport.sleep(self.config.dt)
                if self.transport.is_shutdown():
                    break

        # Wait for first pose
        self.transport.wait_for_pose()

        # Reset simulation if configured
        if self.config.reset.use_service:
            self.transport.reset_simulation(
                num_calls=self.config.reset.num_calls,
                sleep_between=self.config.reset.sleep_between_s,
            )

        # Sleep to let simulator settle and collect fresh pose data
        self.transport.sleep(self.config.reset.sleep_between_s)

        # Acquire target
        self._target_pose = self._acquire_target()
        logger.info(
            "Episode %d: target at [%.2f, %.2f, %.1f°]",
            self._episode,
            *self._target_pose,
        )

        # Publish initial zero actuator commands
        self._calculate_actuator_inputs(self._norm_action)
        self.transport.publish_actuator_setpoints(self._thrusters, self._angles)

        # Collect first observation
        obs = self._get_obs()
        norm_obs = self._normalize_obs(obs)
        info = self._get_info()

        return norm_obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._time_step += 1

        # Save previous state
        self._eta_prev[:] = self._eta
        self._action_prev[:] = self._action
        self._norm_action_prev[:] = self._norm_action
        self._obs_time_prev = self._obs_time
        self._thrusters_prev[:] = self._thrusters
        self._angles_prev[:] = self._angles

        # Apply new action
        self._norm_action = action.copy()
        self._action = action.copy() * vessel.MAX_THRUSTER_RPM
        self._calculate_actuator_inputs(action)
        self.transport.publish_actuator_setpoints(self._thrusters, self._angles)

        # Update waypoint target if configured
        if self.config.target.type == TargetType.waypoint:
            wp = self.transport.get_waypoint()
            if wp is not None:
                self._target_pose[:] = wp

        # Sleep for control period
        self.transport.sleep(self.config.dt)

        # Legacy render-equivalent sleep (models trained with rendering active
        # experienced ~250ms per step from clock.tick(4); this preserves that
        # timing for backward compatibility when evaluating legacy models)
        if self.config.legacy_step_sleep > 0:
            self.transport.sleep(self.config.legacy_step_sleep)

        # Collect observation
        obs = self._get_obs()
        norm_obs = self._normalize_obs(obs)
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        reward = self._compute_reward()
        info = self._get_info()

        # Publish state for standalone viewer (non-blocking)
        self.transport.publish_env_state(
            {
                "epsilon": self._epsilon.tolist(),
                "epsilon_ned": self._epsilon_ned.tolist(),
                "est_velocity": self._est_nu.tolist(),
                "target_pose": self._target_pose.tolist(),
                "thrusters": self._thrusters.tolist(),
                "angles": self._angles.tolist(),
                "reward_components": [
                    self._R_gauss,
                    self._R_AS_gauss,
                    self._R_vel,
                    self._R_thrust,
                    self._R_thrust_d,
                    self._R_angle_d,
                ],
                "reward_total": float(np.squeeze(reward)),
                "time_step": self._time_step,
                "episode": self._episode,
            }
        )

        return norm_obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass  # Transport cleanup handled externally

    # ------------------------------------------------------------------
    # Target acquisition
    # ------------------------------------------------------------------

    def _acquire_target(self) -> np.ndarray:
        if self.config.target.type == TargetType.waypoint:
            wp = self.transport.get_waypoint()
            if wp is not None:
                return np.array(wp)
            # Fallback: hold current position
            return np.array(self.transport.get_pose())

        # Random target
        bounds = np.array(self.config.target.bounds)
        offset = self.np_random.uniform(-1, 1, size=3) * bounds
        return offset

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # Read pose from transport
        x, y, psi_deg = self.transport.get_pose()
        self._eta[:] = [x, y, psi_deg]
        self._obs_time = self.transport.get_timestamp()

        # Position error (NED frame)
        self._epsilon_ned[:] = self._target_pose - self._eta
        self._epsilon_ned[2] = ssa(self._target_pose[2] - self._eta[2])

        # Rotate to body frame
        R = rotate_ned2body(self._eta[2])
        epsilon_xy_body = R @ self._epsilon_ned[:2]
        self._epsilon[:] = [
            epsilon_xy_body[0],
            epsilon_xy_body[1],
            self._epsilon_ned[2],
        ]

        # Velocity estimation
        vel = self.transport.get_velocity()
        if vel is not None:
            self._est_nu[:] = vel
        elif self._obs_time_prev is not None:
            dt = self._obs_time - self._obs_time_prev
            if dt > 0:
                nu_ned = (self._eta - self._eta_prev) / dt
                nu_ned[2] = ssa(self._eta[2] - self._eta_prev[2]) / dt
                u, v = R @ nu_ned[:2]
                self._est_nu[:] = [u, v, nu_ned[2]]

        self._observation = np.concatenate(
            (self._epsilon, self._est_nu, self._action_prev)
        )
        return self._observation

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        eps_x = np.clip(obs[0] / vessel.MAX_DISTANCE, -1, 1)
        eps_y = np.clip(obs[1] / vessel.MAX_DISTANCE, -1, 1)
        eps_psi = obs[2] / vessel.MAX_HEADING_ANGLE
        est_u = np.clip(obs[3] / vessel.MAX_LINEAR_SPEED, -1, 1)
        est_v = np.clip(obs[4] / vessel.MAX_LINEAR_SPEED, -1, 1)
        est_r = np.clip(obs[5] / vessel.MAX_ANGULAR_SPEED, -1, 1)

        state = np.array([eps_x, eps_y, eps_psi, est_u, est_v, est_r])
        self._norm_observation = np.concatenate((state, self._norm_action_prev))
        return self._norm_observation

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        d = np.sqrt(self._epsilon[0] ** 2 + self._epsilon[1] ** 2)
        epsilon_psi = self._epsilon[2]
        assert self._norm_observation is not None
        norm_u, norm_v, norm_r = self._norm_observation[3:6]
        rc = self.config.rewards

        self._R_gauss = gaussian_reward(d, epsilon_psi, self._inv_sigma, rc.w_gauss)
        self._R_AS_gauss = gaussian_reward(
            d, epsilon_psi, self._inv_sigma_AS, rc.w_AS_gauss
        )
        self._R_vel = velocity_penalty(norm_u, norm_v, norm_r, rc.velocity_weight)
        self._R_thrust = thrust_penalty(
            self._thrusters, vessel.MAX_THRUSTER_RPM, rc.thrust_weight
        )
        self._R_thrust_d = thrust_rate_penalty(
            self._thrusters,
            self._thrusters_prev,
            vessel.MAX_THRUSTER_RPM,
            rc.thrust_rate_weight,
        )

        if rc.angle_rate_weight > 0:
            self._R_angle_d = angle_rate_penalty(
                self._angles,
                self._angles_prev,
                vessel.MAX_AZIMUTH_ANGLE,
                rc.angle_rate_weight,
            )
        else:
            self._R_angle_d = 0.0

        termination = rc.termination_penalty if self._terminated_flag else 0.0

        reward = (
            (self._R_gauss + self._R_AS_gauss) / 1.4
            + self._R_vel
            + self._R_thrust
            + self._R_thrust_d
            + self._R_angle_d
            + termination
        )
        return float(np.squeeze(reward))

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def _check_terminated(self) -> bool:
        d = np.sqrt(self._epsilon[0] ** 2 + self._epsilon[1] ** 2)
        if d > vessel.MAX_DISTANCE:
            logger.info("Terminated: distance %.2f m > %.1f m", d, vessel.MAX_DISTANCE)
            self._terminated_flag = True
            return True
        return False

    def _check_truncated(self) -> bool:
        if self._time_step >= self.config.max_time_steps:
            if self._time_step == self.config.max_time_steps:
                logger.info("Truncated at step %d", self._time_step)
            return True
        return False

    # ------------------------------------------------------------------
    # Actuator mapping
    # ------------------------------------------------------------------

    def _calculate_actuator_inputs(self, action: np.ndarray) -> None:
        """Convert 8-dim action (4 x [x, y] components) to thrust RPM + angles."""
        for i in range(4):
            x = action[i * 2]
            y = action[i * 2 + 1]

            if x == 0.0 and y == 0.0:
                self._thrusters[i] = 0.0
                self._angles[i] = self._angles_prev[i]
                continue

            angle = np.arctan2(y, x)
            # Handle -pi edge case for thruster 2
            if angle == np.pi and i == 1:
                angle = -np.pi

            thrust = np.clip(np.sqrt(x**2 + y**2), 0.0, 1.0)
            self._angles[i] = angle / np.pi * 180.0
            self._thrusters[i] = thrust * vessel.MAX_THRUSTER_RPM

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self) -> dict:
        return {
            "observation": self._observation,
            "target_pose": self._target_pose.copy(),
            "thrusters": self._thrusters.copy(),
            "angles": self._angles.copy(),
            "epsilon": self._epsilon.copy(),
            "est_velocity": self._est_nu.copy(),
            "reward_components": {
                "R_gauss": self._R_gauss,
                "R_AS_gauss": self._R_AS_gauss,
                "R_vel": self._R_vel,
                "R_thrust": self._R_thrust,
                "R_thrust_d": self._R_thrust_d,
                "R_angle_d": self._R_angle_d,
            },
            "time_step": self._time_step,
            "episode_count": self._episode,
        }
