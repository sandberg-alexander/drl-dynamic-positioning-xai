import rospy
import torch
import pygame
import time
import pandas as pd
import os
import shap
import signal
import sys
import numpy as np
import math
import glob
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from render_explaination import RenderExplaination
from custom_ros_msgs.msg import ObservationActuatorRefPair, Mode
import milliAmpere1ROS_env

class Agent():
    def __init__(self):
        self.obs = None
        self.action = None
        self.target_pose = None
        self.mode_flag = 0

        self.pub_mode = rospy.Publisher(
            '/drl/mode',
            Mode,
            queue_size=1
        )
        self.mode_msg = Mode()

        #rospy.init_node('xai', anonymous=True)
        rospy.Subscriber('drl/observation_actuator_ref_pair', ObservationActuatorRefPair, self._callback)
        rospy.Subscriber('/drl/mode', Mode, self._mode_callback)

    def _mode_callback(self, msg: Mode):
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
            data.n_y4d_prev
        ]

        self.actuator_ref = [
            (data.n_d1/900, data.alpha_d1),
            (data.n_d2/900, data.alpha_d2),
            (data.n_d3/900, data.alpha_d3),
            (data.n_d4/900, data.alpha_d4),
        ]

        self.target_pose = (data.target_x, data.target_y, data.target_heading)


    def get_observations(self):
        return self.obs
    
    def get_actuator_ref(self):
        return self.actuator_ref
    
    def get_target_pose(self):
        return self.target_pose
    
class Obs2ActionWrapper(torch.nn.Module):
    def __init__(self, model):
        super(Obs2ActionWrapper, self).__init__()
        self.mlp_extractor = model.policy.mlp_extractor.policy_net
        self.action_net = model.policy.action_net

    def forward(self, obs):
        x = self.mlp_extractor(obs)
        action = self.action_net(x)
        return action

class Obs2ValueWrapper(torch.nn.Module):
    def __init__(self, model):
        super(Obs2ValueWrapper, self).__init__()
        self.features_extractor = model.policy.features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.value_net = model.policy.value_net

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        value = self.value_net(latent_vf)
        return value
    



def signal_handler(sig, frame):
    print("\nForced shutdown initiated. Cleaning up...")
    if not rospy.is_shutdown():
        rospy.signal_shutdown("Keyboard interupt")

    try:
        rospy.sleep(0.5)
    except:
        pass

    pygame.quit()

    time.sleep(2)
    os._exit(0)



def ssa_rad(angle):
    return (angle+np.pi) % (2*np.pi) - np.pi

def ssa_alt(angle):
        if angle == -np.pi:
            return np.pi
        return (angle+np.pi) % (2*np.pi) - np.pi

def combine_actuator_ref(actuator_ref, actuator_pos):
    thrust_x = 0
    thrust_y = 0
    tot_angular_thrust = 0
    for i, ((thrust, angle), (x,y)) in enumerate(zip(actuator_ref, actuator_pos)):
        rad = np.deg2rad(90*(i+1)-angle)
        angle = angle * np.pi/180

        thrust_x += thrust * math.cos(angle)
        thrust_y += thrust * math.sin(angle)

        if x*y < 0:
            ang_thrust = abs(x) * thrust * np.cos(rad) + abs(y) * thrust * np.sin(rad)
        else:
            ang_thrust = abs(x) * thrust * np.sin(rad) + abs(y) * thrust * np.cos(rad)
        tot_angular_thrust += ang_thrust

    tot_thrust = math.hypot(thrust_x, thrust_y)
    tot_angle = math.atan2(thrust_y, thrust_x) * 180/np.pi

    return tot_thrust, tot_angle, tot_angular_thrust



def main():
    print("""
    ### ___T_ ################################################
       | n n |                   _     __  __    _    ___ 
       |__E__|      _ __ ___    / \    \ \/ /   / \  |_ _|
    >===]__o[===<  | '_ ` _ \  / _ \    \  /   / _ \  | | 
        [o__]      | | | | | |/ ___ \   /  \  / ___ \ | | 
        /7 [|      |_| |_| |_/_/   \_\ /_/\_\/_/   \_\___| v.0
      \/7  [|_     shap_explaination.py                                           
    ##########################################################

    Starting a XAI run in clean environment ...
    """)

    # Signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Define models and environments
    model = PPO.load("/app/models/training_20250404_165037/models/best_model.zip")
    env = gym.make("milliAmpere1ROS_env/MilliAmpere1ROS-v4", render_mode='human', max_time_steps=200)
    action_low = env.action_space.low
    action_high = env.action_space.high
    obs2action_model = Obs2ActionWrapper(model)
    obs2value_model = Obs2ValueWrapper(model)

    # Define background and sample observations
    # Make sure folder is made and mounted, also explore if our samples should only be leagal configurations
    # could also see if base gets closer to 0 if do symetric positions, rather than +-. Example add 180
    # degrees to heading insted of - heding. 
    num_random = 500
    random_obs = np.array([env.observation_space.sample() for _ in range(num_random)])
    random_obs = np.concatenate([random_obs, -random_obs])
    bakground_obs = random_obs

    #df = pd.read_csv("/app/samples/training_20250328_131443/samples.csv")
    #samples_obs = df.iloc[:,:].values
    samples_obs = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

    bakground_torch = torch.tensor(bakground_obs, dtype=torch.float32)
    samples_torch = torch.tensor(samples_obs, dtype=torch.float32)

    # SHAP explainers
    explainer_action = shap.DeepExplainer(obs2action_model, bakground_torch)
    explainer_value = shap.DeepExplainer(obs2value_model, samples_torch)
    #explainer_value = shap.DeepExplainer(obs2value_model, samples_torch)
    base_vectors = explainer_action.expected_value
    print("Base vectors:", base_vectors)
    # constraints
    max_distance = 10
    max_heading_angle = 180
    max_linear_speed = 3.24
    max_angular_speed = 112.6
    max_thruster_rpm = 900
    max_azimuth_angle = 180

    thruster_x = 1.8
    thruster_y = 0.8
    actuator_pos = np.array([[thruster_x, -thruster_y],
                             [thruster_x, thruster_y],
                             [-thruster_x, thruster_y],
                             [-thruster_x, -thruster_y]])

    # init
    render = RenderExplaination()
    fps = 5
    clock = pygame.time.Clock()
    sleep_time = 0.1
    agent = Agent()

    ros_master_uri = os.environ.get('ROS_MASTER_URI')
    if ros_master_uri == 'http://simulator_local:11311':
        data_path = '/app/runs/sim/'
    else:
        data_path = '/app/runs/real/'
    screen = pygame.display.get_surface()  # or whatever size you use
    width, height = screen.get_size()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')       # H.264 in an .mp4 file
        
    rospy.sleep(sleep_time*3)

    # MAIN LOOP
    try:
        while not rospy.is_shutdown():
            start=time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("event quit")
                    rospy.signal_shutdown("User closed window")
                # Keyboard defined inputs
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_5:
                        agent.mode_msg.mode = 5
                        agent.mode_flag = 5
                        agent.pub_mode.publish(agent.mode_msg)
                    elif event.key == pygame.K_4:
                        agent.mode_msg.mode = 4
                        agent.mode_flag = 4
                        agent.pub_mode.publish(agent.mode_msg)
                    elif event.key == pygame.K_3:
                        agent.mode_msg.mode = 3
                        agent.mode_flag = 3
                        agent.pub_mode.publish(agent.mode_msg)
                        out_fname = f"{data_path}video/{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        video_writer = cv2.VideoWriter(out_fname, fourcc, fps, (width, height))
                    elif event.key == pygame.K_2:
                        agent.mode_msg.mode = 2
                        agent.mode_flag = 2
                        agent.pub_mode.publish(agent.mode_msg)
                        out_fname = f"{data_path}video/{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        video_writer = cv2.VideoWriter(out_fname, fourcc, fps, (width, height))
                    elif event.key == pygame.K_1:
                        agent.mode_msg.mode = 1
                        agent.mode_flag = 1
                        agent.pub_mode.publish(agent.mode_msg)
                        out_fname = f"{data_path}video/{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        video_writer = cv2.VideoWriter(out_fname, fourcc, fps, (width, height))
                    elif event.key == pygame.K_0:
                        agent.mode_msg.mode = 0
                        agent.mode_flag = 0
                        agent.pub_mode.publish(agent.mode_msg)
            
            if agent.mode_msg.mode == 5 and agent.mode_flag == 0:
                latest = max(glob.glob("/app/xai_samples/value_function/sample_*.csv"), key=os.path.getmtime)
                df = pd.read_csv(latest)
                samples_obs = df.iloc[:,:].values
                samples_torch = torch.tensor(samples_obs, dtype=torch.float32)
                explainer_value = shap.DeepExplainer(obs2value_model, samples_torch)
                agent.mode_msg.mode = 0
            
            # # get actions
            # actions = agent.get_actions()
            # while actions is None:
            #     print("Waiting for actions ...")
            #     rospy.sleep(5)
            #     actions = agent.get_actions()

            # actions_list = [
            #     actions.n_x1d, actions.n_y1d,
            #     actions.n_x2d, actions.n_y2d,
            #     actions.n_x3d, actions.n_y3d,
            #     actions.n_x4d, actions.n_y4d
            # ]

            # get target heading
            target_pose = agent.get_target_pose()

            # get actuator ref
            actuator_ref = agent.get_actuator_ref()
            while actuator_ref is None:
                print("Waiting for actuator_ref ...")
                rospy.sleep(5)
                actuator_ref = agent.get_actuator_ref()

            # get observations
            obs = agent.get_observations()
            while obs is None:
                print("Waiting for observations ...")
                rospy.sleep(5)
                obs = agent.get_observations()

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            shap_values_action = explainer_action.shap_values(obs_tensor)
            shap_values_value = explainer_value.shap_values(obs_tensor)
            
            # print(shap_values_value)
            # print('')
            # calculate render arguments
            shap_values_action_list = [arr.flatten().tolist() for arr in shap_values_action]

            shap_values_value_list = [arr.flatten().tolist() for arr in shap_values_value][0]
            # print(shap_values_value_list)
            # print('')
            # print('')
            tot_thrust, tot_angle, tot_angular_thrust = combine_actuator_ref(actuator_ref, actuator_pos)
            x_tilde = obs[0] * max_distance
            y_tilde = obs[1] * max_distance
            psi_tilde = obs[2] * max_heading_angle
            u_hat = obs[3] * max_linear_speed*2
            v_hat = obs[4] * max_linear_speed*2
            r_hat = obs[5] * 360*2

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
                    target_pose
                )
            except pygame.error as e:
                print("Pygame error during rendering:", e)
                break
            
            if agent.mode_msg.mode in [1,2,3]:
                surface = pygame.display.get_surface()
                frame = pygame.surfarray.array3d(surface)          # (W, H, 3) in RGB
                frame = np.transpose(frame, (1, 0, 2))             # -> (H, W, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)      # OpenCV wants BGR

                video_writer.write(frame)                          # add frame to the file
                
                if agent.mode_flag == 0:
                    video_writer.release()        # very important – flushes and closes the file
                    agent.mode_msg.mode = 0

            try:
                rospy.sleep(sleep_time)
            except rospy.ROSInterruptException:
                break
            end=time.perf_counter()
            print(f"{end-start} s")
    
    except Exception as e:
        print(f"Exception in main loop: {e}")
    finally:
        print("Quitting pygame...")
        pygame.quit
        video_writer.release()        # very important – flushes and closes the file


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Exception occurred:", e)
        pygame.quit()
        sys.exit(1)
        
