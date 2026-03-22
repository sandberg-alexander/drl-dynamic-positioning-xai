"""DRL deployer state machine for real-time dynamic positioning."""

from __future__ import annotations

import os
from datetime import datetime
from enum import IntEnum

import numpy as np

from milliampere_drl.config import DeployConfig
from milliampere_drl.spline import build_spline, heading_from_derivative


class DeployMode(IntEnum):
    """Operating modes for the DRL deployer, matching Mode.msg integer values."""

    DP = 0
    DP_TEST = 1
    NORTH_TEST = 2
    SPLINE_TEST = 3
    ACTION_SAMPLE = 4
    VF_SAMPLE = 5


class DRLDeployer:
    """ROS-based DRL deployer with mode-switching state machine.

    Lazy-imports rospy and ROS messages in __init__ so that
    this module can be imported on host without ROS installed.
    """

    def __init__(self, config: DeployConfig, device: str = "cpu") -> None:
        # Lazy-import ROS dependencies
        import gymnasium as gym
        import milliampere_env  # noqa: F401 -- registers MilliAmpere1-v1
        import rospy
        from custom_msgs.msg import NorthEastHeading
        from custom_ros_msgs.msg import Mode, ObservationActuatorRefPair
        from stable_baselines3 import PPO

        self._rospy = rospy
        self._Mode = Mode
        self._ObservationActuatorRefPair = ObservationActuatorRefPair
        self._NorthEastHeading = NorthEastHeading
        self._config = config

        # State
        self.mode_flag = DeployMode.DP
        self.time_step = 0
        self.data: np.ndarray | None = None
        self.pose_nr = 0
        self.init_pose: np.ndarray | None = None
        self.sample_nr = 0

        # Setup publishers
        self.pub_obs_act = rospy.Publisher(
            "/drl/observation_actuator_ref_pair",
            ObservationActuatorRefPair,
            queue_size=1,
        )
        self.pub_target = rospy.Publisher(
            "guidance/waypoint", NorthEastHeading, queue_size=1
        )
        self.pub_mode = rospy.Publisher("/drl/mode", Mode, queue_size=1)
        self.obs_act_msg = ObservationActuatorRefPair()
        self.target_msg = NorthEastHeading()
        self.mode_msg = Mode()

        # Subscribe to mode updates
        rospy.Subscriber("/drl/mode", Mode, self._mode_callback)

        # Load model & environment
        from milliampere_env.milliampere_env import MilliAmpereEnv

        self.model = PPO.load(config.model_path, device=device)
        self.env = gym.make("MilliAmpere1-v1", config_path=config.env_config)
        self.model.set_env(self.env)
        self._dp_env: MilliAmpereEnv = self.env.unwrapped  # type: ignore[assignment]

        # Determine data path
        ros_master_uri = os.environ.get("ROS_MASTER_URI")
        if ros_master_uri == "http://simulator_local:11311":
            self.data_path = config.data_path_sim
        else:
            self.data_path = config.data_path_real
        self.header = (
            "x,y,psi,u,v,r,"
            "n1x,n1y,n2x,n2y,n3x,n3y,n4x,n4y,"
            "n1d,alpha1d,n2d,alpha2d,n3d,alpha3d,n4d,alpha4d"
        )
        self.header2 = "x,y,psi,u,v,r,n1x,n1y,n2x,n2y,n3x,n3y,n4x,n4y"

        # Test parameters (from config)
        self.data_points = 22
        self.spline_x, self.spline_y = build_spline()
        self.s = 0.0

        # Initialize env
        self.obs, _ = self.env.reset()

    def _mode_callback(self, msg) -> None:
        new_mode = msg.mode
        if new_mode != self.mode_flag:
            self.mode_flag = new_mode
            if new_mode == DeployMode.DP:
                self._set_mode(DeployMode.DP)
                self._rospy.loginfo("----- DP mode")
            elif new_mode == DeployMode.DP_TEST:
                self._start_dp_test()
            elif new_mode == DeployMode.NORTH_TEST:
                self._start_north_test()
            elif new_mode == DeployMode.SPLINE_TEST:
                self._start_spline_test()
            elif new_mode == DeployMode.ACTION_SAMPLE:
                self._start_action_sample()
            elif new_mode == DeployMode.VF_SAMPLE:
                self._start_vf_sample()

    def spin(self) -> None:
        while not self._rospy.is_shutdown():
            self.time_step += 1

            if self.mode_flag == DeployMode.DP:
                pass
            elif self.mode_flag == DeployMode.DP_TEST:
                self._run_dp_test()
            elif self.mode_flag == DeployMode.NORTH_TEST:
                self._run_north_test()
            elif self.mode_flag == DeployMode.SPLINE_TEST:
                self._run_spline_test()
            elif self.mode_flag == DeployMode.ACTION_SAMPLE:
                self._run_action_sample()
            elif self.mode_flag == DeployMode.VF_SAMPLE:
                self._run_vf_sample()

            action, _ = self.model.predict(self.obs, deterministic=True)
            obs_next, reward, terminated, truncated, info = self.env.step(action)

            self._publish_obs_act(self.obs)
            self._log_data()
            self.obs = obs_next

    def _set_mode(self, mode: DeployMode) -> None:
        self.mode_flag = mode
        self.mode_msg.mode = int(mode)
        self.pub_mode.publish(self.mode_msg)
        self.time_step = 1

    def _start_dp_test(self) -> None:
        self._set_mode(DeployMode.DP_TEST)
        self._rospy.loginfo("----- Testing mode: DP")
        self.pose_nr = 0
        self.init_pose = self._dp_env.target_pose
        cfg = self._config
        self.data = np.zeros((cfg.vep_length_dp * cfg.test_length_dp, self.data_points))

    def _run_dp_test(self) -> None:
        cfg = self._config
        if self.time_step >= cfg.test_length_dp * cfg.vep_length_dp + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}data_test_dp_{ts}.csv"
            assert self.data is not None
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header,
                comments="",
            )
            self._set_mode(DeployMode.DP)
            self._rospy.loginfo("----- DP mode")
        elif self.time_step % cfg.vep_length_dp == 0:
            if self.pose_nr < len(cfg.test_pose):
                assert self.init_pose is not None
                dx, dy, heading = cfg.test_pose[self.pose_nr]
                self.target_msg.north = self.init_pose[0] + dx
                self.target_msg.east = self.init_pose[1] + dy
                self.target_msg.heading = heading
                self.pub_target.publish(self.target_msg)
                self.pose_nr += 1

    def _start_north_test(self) -> None:
        self._set_mode(DeployMode.NORTH_TEST)
        self._rospy.loginfo("----- Testing mode: path-following-north")
        self.target_msg.heading = 0
        self.data = np.zeros((self._config.test_length_north, self.data_points))

    def _run_north_test(self) -> None:
        cfg = self._config
        if self.time_step >= cfg.test_length_north + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}data_test_north_{ts}.csv"
            assert self.data is not None
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header,
                comments="",
            )
            self._set_mode(DeployMode.DP)
            self._rospy.loginfo("----- DP mode")
        else:
            self.target_msg.north = self._dp_env.target_pose[0] + cfg.ds_north
            self.pub_target.publish(self.target_msg)

    def _start_spline_test(self) -> None:
        self._set_mode(DeployMode.SPLINE_TEST)
        self._rospy.loginfo("----- Testing mode: path-following-spline")
        self.init_pose = self._dp_env.target_pose
        self.data = np.zeros((self._config.test_length_spline, self.data_points))
        self.s = 0.0

    def _run_spline_test(self) -> None:
        cfg = self._config
        if self.time_step >= cfg.test_length_spline + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}data_test_spline_{ts}.csv"
            assert self.data is not None
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header,
                comments="",
            )
            self._set_mode(DeployMode.DP)
            self._rospy.loginfo("----- DP mode")
        else:
            assert self.init_pose is not None
            self.s += cfg.ds_spline
            x_t = float(self.spline_x(self.s))
            y_t = float(self.spline_y(self.s))
            dx_dt = float(self.spline_x(self.s, 1))
            dy_dt = float(self.spline_y(self.s, 1))
            psi_t = heading_from_derivative(dx_dt, dy_dt)
            self.target_msg.north = self.init_pose[0] + x_t
            self.target_msg.east = self.init_pose[1] + y_t
            self.target_msg.heading = psi_t
            self.pub_target.publish(self.target_msg)

    def _start_action_sample(self) -> None:
        self._set_mode(DeployMode.ACTION_SAMPLE)
        self._rospy.loginfo("----- Sampling mode: action")
        self.pose_nr = 0
        self.init_pose = self._dp_env.target_pose
        cfg = self._config
        self.data = np.zeros(
            (cfg.sample_length_action * cfg.vep_length_action // 2, len(self.obs))
        )

    def _run_action_sample(self) -> None:
        cfg = self._config
        if self.time_step >= cfg.sample_length_action * cfg.vep_length_dp + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{cfg.xai_action_sample_path}sample_{ts}.csv"
            assert self.data is not None
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header2,
                comments="",
            )
            self._set_mode(DeployMode.DP)
            self._rospy.loginfo("----- DP mode")
        elif self.time_step % cfg.vep_length_action == 0:
            if self.pose_nr < len(cfg.sample_pose):
                assert self.init_pose is not None
                dx, dy, heading = cfg.sample_pose[self.pose_nr]
                self.target_msg.north = self.init_pose[0] + dx
                self.target_msg.east = self.init_pose[1] + dy
                self.target_msg.heading = heading
                self.pub_target.publish(self.target_msg)
                self.pose_nr += 1

    def _start_vf_sample(self) -> None:
        self._set_mode(DeployMode.VF_SAMPLE)
        self._rospy.loginfo("----- Sampling mode: value-function")
        self.data = np.zeros((self._config.sample_length_vf, len(self.obs)))

    def _run_vf_sample(self) -> None:
        cfg = self._config
        if self.time_step >= cfg.sample_length_vf + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{cfg.xai_vf_sample_path}sample_{ts}.csv"
            assert self.data is not None
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header2,
                comments="",
            )
            self._rospy.loginfo("Updated acceptable behavior")
            self._set_mode(DeployMode.DP)
            self._rospy.loginfo("----- DP mode")

    def _publish_obs_act(self, obs) -> None:
        m = self.obs_act_msg
        m.x_tilde, m.y_tilde, m.psi_tilde = obs[0], obs[1], obs[2]
        m.u_hat, m.v_hat, m.r_hat = obs[3], obs[4], obs[5]
        m.n_x1d_prev, m.n_y1d_prev = obs[6], obs[7]
        m.n_x2d_prev, m.n_y2d_prev = obs[8], obs[9]
        m.n_x3d_prev, m.n_y3d_prev = obs[10], obs[11]
        m.n_x4d_prev, m.n_y4d_prev = obs[12], obs[13]
        thrusters = self._dp_env.thrusters
        angles = self._dp_env.angles
        m.n_d1, m.alpha_d1 = thrusters[0], angles[0]
        m.n_d2, m.alpha_d2 = thrusters[1], angles[1]
        m.n_d3, m.alpha_d3 = thrusters[2], angles[2]
        m.n_d4, m.alpha_d4 = thrusters[3], angles[3]
        target = self._dp_env.target_pose
        m.target_x = target[0]
        m.target_y = target[1]
        m.target_heading = target[2]
        epsilon_ned = self._dp_env.epsilon_ned
        m.x_ned_err = epsilon_ned[0]
        m.y_ned_err = epsilon_ned[1]
        m.psi_ned_err = epsilon_ned[2]
        m.time_step = self.time_step
        self.pub_obs_act.publish(m)

    def _log_data(self) -> None:
        m = self.obs_act_msg
        if self.mode_flag in (
            DeployMode.DP_TEST,
            DeployMode.NORTH_TEST,
            DeployMode.SPLINE_TEST,
        ):
            assert self.data is not None
            self.data[self.time_step - 1] = [
                m.x_tilde,
                m.y_tilde,
                m.psi_tilde,
                m.u_hat,
                m.v_hat,
                m.r_hat,
                m.n_x1d_prev,
                m.n_y1d_prev,
                m.n_x2d_prev,
                m.n_y2d_prev,
                m.n_x3d_prev,
                m.n_y3d_prev,
                m.n_x4d_prev,
                m.n_y4d_prev,
                m.n_d1,
                m.alpha_d1,
                m.n_d2,
                m.alpha_d2,
                m.n_d3,
                m.alpha_d3,
                m.n_d4,
                m.alpha_d4,
            ]
        elif (
            self.mode_flag == DeployMode.ACTION_SAMPLE and (self.time_step - 1) % 2 == 0
        ):
            assert self.data is not None
            self.data[self.sample_nr] = self.obs
            self.sample_nr += 1
        elif self.mode_flag == DeployMode.VF_SAMPLE:
            assert self.data is not None
            self.data[self.time_step - 1] = self.obs
