import milliampere_env  # noqa: F401 -- registers MilliAmpere1-v1
from milliampere_dp import __version__
import os
import math
import rospy
import numpy as np
import gymnasium as gym
from datetime import datetime
from enum import IntEnum
from stable_baselines3 import PPO
from scipy.interpolate import CubicSpline

from custom_msgs.msg import NorthEastHeading
from custom_ros_msgs.msg import ObservationActuatorRefPair, Mode


class DeployMode(IntEnum):
    """Operating modes for the DRL deployer, matching Mode.msg integer values."""

    DP = 0
    DP_TEST = 1
    NORTH_TEST = 2
    SPLINE_TEST = 3
    ACTION_SAMPLE = 4
    VF_SAMPLE = 5


DEFAULT_MODEL_PATH = "/app/models/training_20250404_165037/models/best_model.zip"
DEFAULT_CONFIG_PATH = "/app/configs/env/legacy/v10_equivalent.yaml"

########################################################################
# Utility functions                                                     #
########################################################################


def build_spline():
    """Return *open* cubic splines (x(s), y(s))."""
    control_pts = np.array(
        [
            [0.0, 0.0],  # p0
            [5.0, 2.0],  # p1
            [10.0, 0.0],  # p2
            [15.0, -3.0],  # p3
            [20.0, 0.0],  # p4
        ]
    )
    t = np.linspace(0.0, 1.0, len(control_pts))
    spline_x = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    spline_y = CubicSpline(t, control_pts[:, 1], bc_type="natural")
    return spline_x, spline_y


def heading_from_derivative(dx: float, dy: float) -> float:
    """Return yaw [rad] along derivative (dx, dy)."""
    return math.atan2(dy, dx)


########################################################################
# Main                                                                  #
########################################################################


class DRLDeployer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("drl_deployer", anonymous=True)

        # State
        self.mode_flag = DeployMode.DP
        self.time_step = 0
        self.data = None
        self.pose_nr = 0
        self.init_pose = None

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
        self.model = PPO.load(DEFAULT_MODEL_PATH)
        self.env = gym.make(
            "MilliAmpere1-v1",
            config_path=DEFAULT_CONFIG_PATH,
        )
        self.model.set_env(self.env)

        # Determine data path
        ros_master_uri = os.environ.get("ROS_MASTER_URI")
        if ros_master_uri == "http://simulator_local:11311":
            self.data_path = "/app/runs/sim/"
        else:
            self.data_path = "/app/runs/real/"
        self.header = "x,y,psi,u,v,r,n1x,n1y,n2x,n2y,n3x,n3y,n4x,n4y,n1d,alpha1d,n2d,alpha2d,n3d,alpha3d,n4d,alpha4d"
        self.header2 = "x,y,psi,u,v,r,n1x,n1y,n2x,n2y,n3x,n3y,n4x,n4y"

        # Test parameters
        self.data_points = 22

        self.vep_length_dp = 200
        self.test_length_dp = 5
        self.test_pose = [(4, 0, 0), (4, 4, 0), (0, 0, 0), (0, 0, np.pi)]

        self.test_length_north = 200
        self.ds_north = 0.1

        self.test_length_spline = 200
        self.spline_x, self.spline_y = build_spline()
        self.s = 0.0
        self.ds_spline = 0.005

        self.sample_nr = 0
        self.vep_length_action = 200
        self.sample_length_action = 5
        self.sample_pose = [
            (4, 3, np.deg2rad(32)),
            (3.3, -1, np.deg2rad(-149)),
            (-0.8, -3.4, np.deg2rad(-19)),
            (0, 0, np.deg2rad(27)),
        ]

        self.sample_length_vf = 1000

        # Initialize env
        self.obs, _ = self.env.reset()

    def _mode_callback(self, msg: Mode):
        """Update mode_flag from incoming Mode message."""
        new_mode = msg.mode
        if new_mode != self.mode_flag:
            self.mode_flag = new_mode
            if new_mode == DeployMode.DP:
                self._set_mode(DeployMode.DP)
                rospy.loginfo("----- DP mode")
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

    def spin(self):
        # Mode switching via ROS /drl/mode topic (from XAI dashboard keyboard)
        while not rospy.is_shutdown():
            self.time_step += 1

            # Execute mode logic
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

            # Step the agent
            action, _ = self.model.predict(self.obs, deterministic=True)
            obs_next, reward, terminated, truncated, info = self.env.step(action)

            # Publish obs-act pair
            self._publish_obs_act(self.obs)

            # Log data from current time-step
            self._log_data()

            # Advance
            self.obs = obs_next

    def _set_mode(self, mode: DeployMode):
        self.mode_flag = mode
        self.mode_msg.mode = int(mode)
        self.pub_mode.publish(self.mode_msg)
        self.time_step = 1

    def _start_dp_test(self):
        self._set_mode(DeployMode.DP_TEST)
        rospy.loginfo("----- Testing mode: DP")
        self.pose_nr = 0
        self.init_pose = self.env.unwrapped._target_pose
        self.data = np.zeros(
            (self.vep_length_dp * self.test_length_dp, self.data_points)
        )

    def _run_dp_test(self):
        if self.time_step >= self.test_length_dp * self.vep_length_dp + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}data_test_dp_{ts}.csv"
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header,
                comments="",
            )
            self._set_mode(DeployMode.DP)
            rospy.loginfo("----- DP mode")
        elif self.time_step % self.vep_length_dp == 0:
            if self.pose_nr < 4:
                dx, dy, heading = self.test_pose[self.pose_nr]
                self.target_msg.north = self.init_pose[0] + dx
                self.target_msg.east = self.init_pose[1] + dy
                self.target_msg.heading = heading
                self.pub_target.publish(self.target_msg)
                self.pose_nr += 1

    def _start_north_test(self):
        self._set_mode(DeployMode.NORTH_TEST)
        rospy.loginfo("----- Testing mode: path-following-north")
        self.target_msg.heading = 0
        self.data = np.zeros((self.test_length_north, self.data_points))

    def _run_north_test(self):
        if self.time_step >= self.test_length_north + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}data_test_north_{ts}.csv"
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header,
                comments="",
            )
            self._set_mode(DeployMode.DP)
            rospy.loginfo("----- DP mode")
        else:
            self.target_msg.north = self.env.unwrapped._target_pose[0] + self.ds_north
            self.pub_target.publish(self.target_msg)

    def _start_spline_test(self):
        self._set_mode(DeployMode.SPLINE_TEST)
        rospy.loginfo("----- Testing mode: path-following-spline")
        self.init_pose = self.env.unwrapped._target_pose
        self.data = np.zeros((self.test_length_spline, self.data_points))
        self.s = 0.0

    def _run_spline_test(self):
        if self.time_step >= self.test_length_spline + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_path}data_test_spline_{ts}.csv"
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header,
                comments="",
            )
            self._set_mode(DeployMode.DP)
            rospy.loginfo("----- DP mode")
        else:
            self.s += self.ds_spline
            x_t = float(self.spline_x(self.s))
            y_t = float(self.spline_y(self.s))
            dx_dt = float(self.spline_x(self.s, 1))
            dy_dt = float(self.spline_y(self.s, 1))
            psi_t = heading_from_derivative(dx_dt, dy_dt)
            self.target_msg.north = self.init_pose[0] + x_t
            self.target_msg.east = self.init_pose[1] + y_t
            self.target_msg.heading = psi_t
            self.pub_target.publish(self.target_msg)

    def _start_action_sample(self):
        self._set_mode(DeployMode.ACTION_SAMPLE)
        rospy.loginfo("----- Sampling mode: action")
        self.pose_nr = 0
        self.init_pose = self.env.unwrapped._target_pose
        self.data = np.zeros(
            (self.sample_length_action * self.vep_length_action // 2, len(self.obs))
        )

    def _run_action_sample(self):
        if self.time_step >= self.sample_length_action * self.vep_length_dp + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/app/xai_samples/action/sample_{ts}.csv"
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header2,
                comments="",
            )
            self._set_mode(DeployMode.DP)
            rospy.loginfo("----- DP mode")
        elif self.time_step % self.vep_length_action == 0:
            if self.pose_nr < 4:
                dx, dy, heading = self.sample_pose[self.pose_nr]
                self.target_msg.north = self.init_pose[0] + dx
                self.target_msg.east = self.init_pose[1] + dy
                self.target_msg.heading = heading
                self.pub_target.publish(self.target_msg)
                self.pose_nr += 1

    def _start_vf_sample(self):
        self._set_mode(DeployMode.VF_SAMPLE)
        rospy.loginfo("----- Sampling mode: value-function")
        self.data = np.zeros((self.sample_length_vf, len(self.obs)))

    def _run_vf_sample(self):
        if self.time_step >= self.sample_length_vf + 1:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/app/xai_samples/value_function/sample_{ts}.csv"
            np.savetxt(
                filename,
                self.data,
                delimiter=",",
                fmt="%f",
                header=self.header2,
                comments="",
            )
            rospy.loginfo("Updated acceptable behavior")
            self._set_mode(DeployMode.DP)
            rospy.loginfo("----- DP mode")

    def _publish_obs_act(self, obs):
        (
            self.obs_act_msg.x_tilde,
            self.obs_act_msg.y_tilde,
            self.obs_act_msg.psi_tilde,
        ) = obs[0], obs[1], obs[2]
        self.obs_act_msg.u_hat, self.obs_act_msg.v_hat, self.obs_act_msg.r_hat = (
            obs[3],
            obs[4],
            obs[5],
        )
        self.obs_act_msg.n_x1d_prev, self.obs_act_msg.n_y1d_prev = obs[6], obs[7]
        self.obs_act_msg.n_x2d_prev, self.obs_act_msg.n_y2d_prev = obs[8], obs[9]
        self.obs_act_msg.n_x3d_prev, self.obs_act_msg.n_y3d_prev = obs[10], obs[11]
        self.obs_act_msg.n_x4d_prev, self.obs_act_msg.n_y4d_prev = obs[12], obs[13]
        thrusters = self.env.unwrapped._thrusters
        angles = self.env.unwrapped._angles
        self.obs_act_msg.n_d1, self.obs_act_msg.alpha_d1 = thrusters[0], angles[0]
        self.obs_act_msg.n_d2, self.obs_act_msg.alpha_d2 = thrusters[1], angles[1]
        self.obs_act_msg.n_d3, self.obs_act_msg.alpha_d3 = thrusters[2], angles[2]
        self.obs_act_msg.n_d4, self.obs_act_msg.alpha_d4 = thrusters[3], angles[3]
        self.obs_act_msg.target_x = self.env.unwrapped._target_pose[0]
        self.obs_act_msg.target_y = self.env.unwrapped._target_pose[1]
        self.obs_act_msg.target_heading = self.env.unwrapped._target_pose[2]
        epsilon_ned = self.env.unwrapped._epsilon_ned
        self.obs_act_msg.x_ned_err = epsilon_ned[0]
        self.obs_act_msg.y_ned_err = epsilon_ned[1]
        self.obs_act_msg.psi_ned_err = epsilon_ned[2]
        self.obs_act_msg.time_step = self.time_step
        self.pub_obs_act.publish(self.obs_act_msg)

    def _log_data(self):
        if self.mode_flag in (
            DeployMode.DP_TEST,
            DeployMode.NORTH_TEST,
            DeployMode.SPLINE_TEST,
        ):
            self.data[self.time_step - 1] = [
                self.obs_act_msg.x_tilde,
                self.obs_act_msg.y_tilde,
                self.obs_act_msg.psi_tilde,
                self.obs_act_msg.u_hat,
                self.obs_act_msg.v_hat,
                self.obs_act_msg.r_hat,
                self.obs_act_msg.n_x1d_prev,
                self.obs_act_msg.n_y1d_prev,
                self.obs_act_msg.n_x2d_prev,
                self.obs_act_msg.n_y2d_prev,
                self.obs_act_msg.n_x3d_prev,
                self.obs_act_msg.n_y3d_prev,
                self.obs_act_msg.n_x4d_prev,
                self.obs_act_msg.n_y4d_prev,
                self.obs_act_msg.n_d1,
                self.obs_act_msg.alpha_d1,
                self.obs_act_msg.n_d2,
                self.obs_act_msg.alpha_d2,
                self.obs_act_msg.n_d3,
                self.obs_act_msg.alpha_d3,
                self.obs_act_msg.n_d4,
                self.obs_act_msg.alpha_d4,
            ]
        elif (
            self.mode_flag == DeployMode.ACTION_SAMPLE and (self.time_step - 1) % 2 == 0
        ):
            self.data[self.sample_nr] = self.obs
            self.sample_nr += 1
        elif self.mode_flag == DeployMode.VF_SAMPLE:
            self.data[self.time_step - 1] = self.obs


if __name__ == "__main__":
    print(f"""
    ### ___T_ ################################################
       | n n |                   _      ____  ____  _
       |__E__|      _ __ ___    / \    |  _ \|  _ \| |        
    >===]__o[===<  | '_ ` _ \  / _ \   | | | | |_) | |        
        [o__]      | | | | | |/ ___ \  | |_| |  _ <| |___     
        /7 [|      |_| |_| |_/_/   \_\ |____/|_| \_\_____| v{__version__}
      \/7  [|_     deploy.py                                           
    ##########################################################

    Starting DRL deployment in DP mode ...
    Options:
          '0' - DP              (DeployMode.DP)
          '1' - DP test         (DeployMode.DP_TEST)
          '2' - North following (DeployMode.NORTH_TEST)
          '3' - Spline following(DeployMode.SPLINE_TEST)
          '4' - Action sampling (DeployMode.ACTION_SAMPLE)
          '5' - VF sampling     (DeployMode.VF_SAMPLE)
    """)
    node = DRLDeployer()
    node.spin()
