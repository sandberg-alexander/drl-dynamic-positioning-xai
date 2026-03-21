"""SHAP-based XAI explanation dashboard for milliAmpere1 DRL-DP."""

from __future__ import annotations

import glob
import os
import signal
import sys
import time

import cv2
import numpy as np
import pandas as pd
import pygame
import shap
import torch
from milliampere_dp.vessel import (
    MAX_ANGULAR_SPEED,
    MAX_DISTANCE,
    MAX_HEADING_ANGLE,
    MAX_LINEAR_SPEED,
    MAX_THRUSTER_RPM,
    THRUSTER_ARM_X,
    THRUSTER_ARM_Y,
)

from milliampere_xai.dashboard import RenderExplanation
from milliampere_xai.model_wrappers import (
    Obs2ActionWrapper,
    Obs2ValueWrapper,
    combine_actuator_ref,
)


class Agent:
    """ROS subscriber that receives observations, actions,
    and state from the deployer."""

    def __init__(self):
        import rospy
        from custom_ros_msgs.msg import Mode, ObservationActuatorRefPair

        self._rospy = rospy

        self.obs = None
        self.action = None
        self.actuator_ref = None
        self.target_pose = None
        self.epsilon_ned = None
        self.mode_flag = 0

        self.pub_mode = rospy.Publisher("/drl/mode", Mode, queue_size=1)
        self.mode_msg = Mode()
        self.start_init = time.time()
        self.start = self.start_init

        rospy.Subscriber(
            "drl/observation_actuator_ref_pair",
            ObservationActuatorRefPair,
            self._callback,
        )
        rospy.Subscriber("/drl/mode", Mode, self._mode_callback)

    def _mode_callback(self, msg):
        """Update mode_flag from incoming Mode message."""
        self.mode_flag = msg.mode

    def _callback(self, data):
        self.obs = [
            data.x_tilde,
            data.y_tilde,
            data.psi_tilde,
            data.u_hat,
            data.v_hat,
            data.r_hat,
            data.n_x1d_prev,
            data.n_y1d_prev,
            data.n_x2d_prev,
            data.n_y2d_prev,
            data.n_x3d_prev,
            data.n_y3d_prev,
            data.n_x4d_prev,
            data.n_y4d_prev,
        ]

        self.actuator_ref = [
            (data.n_d1 / MAX_THRUSTER_RPM, data.alpha_d1),
            (data.n_d2 / MAX_THRUSTER_RPM, data.alpha_d2),
            (data.n_d3 / MAX_THRUSTER_RPM, data.alpha_d3),
            (data.n_d4 / MAX_THRUSTER_RPM, data.alpha_d4),
        ]

        self.target_pose = (data.target_x, data.target_y, data.target_heading)
        self.epsilon_ned = (data.x_ned_err, data.y_ned_err, data.psi_ned_err)

        self.time_step = data.time_step
        if data.time_step == 1:
            self.start = time.time()
        elif self.start == self.start_init:
            self.start = self.start - data.time_step * 0.25

    def get_observations(self):
        return self.obs

    def get_actuator_ref(self):
        return self.actuator_ref

    def get_target_pose(self):
        return self.target_pose

    def get_time(self):
        self.end = time.time()
        return int(self.end - self.start)


def _signal_handler(sig, frame):
    """Graceful shutdown on SIGINT/SIGTERM."""
    import rospy

    print("\nForced shutdown initiated. Cleaning up...")
    if not rospy.is_shutdown():
        rospy.signal_shutdown("Keyboard interrupt")
    try:
        rospy.sleep(0.5)
    except Exception:
        pass
    pygame.quit()
    time.sleep(2)
    os._exit(0)


def main():
    import argparse

    import gymnasium as gym
    import milliampere_env  # noqa: F401 -- registers MilliAmpere1-v1
    import rospy
    from stable_baselines3 import PPO

    from milliampere_xai import __version__

    parser = argparse.ArgumentParser(description="XAI SHAP explanation dashboard")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU rendering (default: software rendering)",
    )
    parser.add_argument("--model", default=None, help="Override model path")
    parser.add_argument("--env-config", default=None, help="Override env config path")
    args = parser.parse_args()

    model_path = (
        args.model or "/app/models/training_20250404_165037/models/best_model.zip"
    )
    env_config = args.env_config or "/app/configs/env/legacy/v4_equivalent.yaml"

    if not args.gpu:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

    print(f"""
    ### ___T_ ################################################
       | n n |                   _     __  __    _    ___
       |__E__|      _ __ ___    / \\    \\ \\/ /   / \\  |_ _|
    >===]__o[===<  | '_ ` _ \\  / _ \\    \\  /   / _ \\  | |
        [o__]      | | | | | |/ ___ \\   /  \\  / ___ \\ | |
        /7 [|      |_| |_| |_/_/   \\_\\ /_/\\_\\/_/   \\_\\___| v{__version__}
      \\/7  [|_     xai-explain
    ##########################################################

    Starting a XAI run ({"GPU" if args.gpu else "software"} rendering) ...

    Keyboard controls (in dashboard window):
          '0' - DP mode              (idle dynamic positioning)
          '1' - DP test              (waypoint sequence, records video)
          '2' - North following      (northward path, records video)
          '3' - Spline following     (spline path, records video)
          '4' - Action sampling      (collect SHAP action samples)
          '5' - VF sampling          (collect value function samples)
    """)

    # Init ROS node before creating subscribers (new env no longer does this)
    rospy.init_node("xai", anonymous=True)

    # Signal handler
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Define models and environments
    model = PPO.load(model_path, device="cpu")
    env = gym.make("MilliAmpere1-v1", config_path=env_config)
    action_low = env.action_space.low  # type: ignore[attr-defined]
    action_high = env.action_space.high  # type: ignore[attr-defined]
    obs2action_model = Obs2ActionWrapper(model)
    obs2value_model = Obs2ValueWrapper(model)

    # Define background and sample observations
    # Make sure folder is made and mounted, also explore if
    # our samples should only be legal configurations.
    # Could also see if base gets closer to 0 if do symmetric
    # positions, rather than +-. Example add 180 degrees to
    # heading instead of -heading.
    num_random = 500
    random_obs = np.array([env.observation_space.sample() for _ in range(num_random)])
    random_obs = np.concatenate([random_obs, -random_obs])
    background_obs = random_obs

    samples_obs = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    background_torch = torch.tensor(background_obs, dtype=torch.float32)
    samples_torch = torch.tensor(samples_obs, dtype=torch.float32)

    # SHAP explainers
    explainer_action = shap.DeepExplainer(obs2action_model, background_torch)
    explainer_value = shap.DeepExplainer(obs2value_model, samples_torch)
    base_vectors = explainer_action.expected_value
    print("Base vectors:", base_vectors)
    actuator_pos = np.array(
        [
            [THRUSTER_ARM_X, -THRUSTER_ARM_Y],
            [THRUSTER_ARM_X, THRUSTER_ARM_Y],
            [-THRUSTER_ARM_X, THRUSTER_ARM_Y],
            [-THRUSTER_ARM_X, -THRUSTER_ARM_Y],
        ]
    )

    # init pygame (font subsystem needs explicit init before creating windows)
    pygame.init()
    render = RenderExplanation()
    fps = 5
    clock = pygame.time.Clock()
    sleep_time = 0.1
    agent = Agent()

    ros_master_uri = os.environ.get("ROS_MASTER_URI")
    if ros_master_uri == "http://simulator_local:11311":
        data_path = "/app/runs/sim/"
    else:
        data_path = "/app/runs/real/"
    # Video recording setup (actual surface size captured when writer is created)
    video_dir = f"{data_path}video"
    os.makedirs(video_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    video_ext = ".avi"
    video_writer = None

    rospy.sleep(sleep_time * 3)

    # MAIN LOOP
    try:
        while not rospy.is_shutdown():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("event quit")
                    rospy.signal_shutdown("User closed window")
                # Keyboard defined inputs
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_5:  # VF_SAMPLE
                        agent.mode_msg.mode = 5
                        agent.mode_flag = 5
                        agent.pub_mode.publish(agent.mode_msg)
                    elif event.key == pygame.K_4:  # ACTION_SAMPLE
                        agent.mode_msg.mode = 4
                        agent.mode_flag = 4
                        agent.pub_mode.publish(agent.mode_msg)
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        mode_num = event.key - pygame.K_0
                        agent.mode_msg.mode = mode_num
                        agent.mode_flag = mode_num
                        agent.pub_mode.publish(agent.mode_msg)
                        # Start video recording with current surface size
                        surf = pygame.display.get_surface()
                        w, h = surf.get_size()
                        out_fname = (
                            f"{video_dir}/{time.strftime('%Y%m%d_%H%M%S')}{video_ext}"
                        )
                        video_writer = cv2.VideoWriter(out_fname, fourcc, fps, (w, h))
                        if not video_writer.isOpened():
                            print(
                                f"Warning: could not open video "
                                f"writer ({w}x{h}) "
                                f"at {out_fname}"
                            )
                            video_writer = None
                        else:
                            print(f"Recording to {out_fname} ({w}x{h})")
                    elif event.key == pygame.K_0:  # DP
                        agent.mode_msg.mode = 0
                        agent.mode_flag = 0
                        agent.pub_mode.publish(agent.mode_msg)

            if agent.mode_msg.mode == 5 and agent.mode_flag == 0:
                latest = max(
                    glob.glob("/app/xai_samples/value_function/sample_*.csv"),
                    key=os.path.getmtime,
                )
                df = pd.read_csv(latest)
                samples_obs = df.iloc[:, :].values
                samples_torch = torch.tensor(samples_obs, dtype=torch.float32)
                explainer_value = shap.DeepExplainer(obs2value_model, samples_torch)
                agent.mode_msg.mode = 0

            # get actuator ref
            actuator_ref = agent.get_actuator_ref()
            while actuator_ref is None:
                print("Waiting for deployer data ...")
                rospy.sleep(5)
                actuator_ref = agent.get_actuator_ref()

            # get observations
            obs = agent.get_observations()
            while obs is None:
                print("Waiting for observations ...")
                rospy.sleep(5)
                obs = agent.get_observations()

            # get target heading (arrives with the same callback as obs/actuator_ref)
            target_pose = agent.get_target_pose()
            while target_pose is None:
                print("Waiting for target pose ...")
                rospy.sleep(1)
                target_pose = agent.get_target_pose()

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            shap_values_action = explainer_action.shap_values(obs_tensor)
            shap_values_value = explainer_value.shap_values(obs_tensor)

            shap_values_action_list = [
                arr.flatten().tolist()  # type: ignore[union-attr]
                for arr in shap_values_action
            ]
            shap_values_value_list = [
                arr.flatten().tolist()  # type: ignore[union-attr]
                for arr in shap_values_value
            ][0]

            tot_thrust, tot_angle, tot_angular_thrust = combine_actuator_ref(
                actuator_ref, actuator_pos
            )
            x_tilde = obs[0] * MAX_DISTANCE
            y_tilde = obs[1] * MAX_DISTANCE
            psi_tilde = obs[2] * MAX_HEADING_ANGLE
            u_hat = obs[3] * MAX_LINEAR_SPEED * 2
            v_hat = obs[4] * MAX_LINEAR_SPEED * 2
            r_hat = obs[5] * MAX_ANGULAR_SPEED * 2

            # cap fps to fps limit
            clock.tick(fps)
            try:
                render.render_frame(
                    shap_values_action_list,
                    shap_values_value_list,
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
                    agent.time_step,
                    agent.get_time(),
                    epsilon_ned=agent.epsilon_ned,
                )
            except pygame.error as e:
                print("Pygame error during rendering:", e)
                break

            if agent.mode_msg.mode in [1, 2, 3]:
                if video_writer is not None:
                    surface = pygame.display.get_surface()
                    frame = pygame.surfarray.array3d(surface)  # (W, H, 3) in RGB
                    frame = np.transpose(frame, (1, 0, 2))  # -> (H, W, 3)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame)

                if agent.mode_flag == 0:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    agent.mode_msg.mode = 0

            try:
                rospy.sleep(sleep_time)
            except rospy.ROSInterruptException:
                break
    except Exception as e:
        print(f"Exception in main loop: {e}")
    finally:
        print("Quitting pygame...")
        pygame.quit()
        if video_writer is not None:
            video_writer.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception occurred:", e)
        pygame.quit()
        sys.exit(1)
