from render_explanation import RenderExplaination
#from test4 import RenderExplaination
import pygame
import math
import rospy
import sys
import signal
import os
import time
from stable_baselines3 import PPO
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import shap
import milliampereROS_env
import shap.models

from drl_msgs.msg import Observation, Action


class Agent():
    def __init__(self):
        self.obs = None
        self.action = None

        rospy.init_node('xai', anonymous=True)
        rospy.Subscriber('/drl/observation', Observation, self._observation_callback)
        rospy.Subscriber('/drl/action', Action, self._action_callback)

    def _observation_callback(self, data):
        self.obs = data
    
    def _action_callback(self, data):
        self.action = data

    def get_observations(self):
        return self.obs
    
    def get_actions(self):
        return self.action
    
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
        self.value_net = model.policy.value_net  # Final linear layer -> scalar

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 1) Extract low-level features from obs
        features = self.features_extractor(obs)
        # 2) Get the (policy_latent, value_latent) from MlpExtractor
        latent_pi, latent_vf = self.mlp_extractor(features)
        # 3) Feed the value latent through the final linear layer to get V(s)
        value = self.value_net(latent_vf)
        return value
    

def combine_vectors(vectors):
    total_x = sum(amplitude * math.cos(angle) for amplitude, angle in vectors)
    total_y = sum(amplitude * math.sin(angle) for amplitude, angle in vectors)

    resultant_magnitude = math.hypot(total_x, total_y)
    resultant_angle = math.atan2(total_y, total_x)
    
    return resultant_magnitude, resultant_angle


def shutdown_hook():
    print("ROS is shutting down. Cleaning up...")


def signal_handler(sig,frame):
    print("\nForced shutdown initiated. Cleaning up...")
    # Force ROS to shut down
    if not rospy.is_shutdown():
        rospy.signal_shutdown("Keyboard interrupt")
    
    # Give ROS a moment to process the shutdown
    try:
        rospy.sleep(0.5)
    except:
        pass
    
    # Ensure Pygame is properly quit
    pygame.quit()
    
    # Force exit if still hanging after 2 seconds
    time.sleep(2)
    os._exit(0)

def main():
    print("""
    #### ___T_ ################################################
        | n n |                   _     __  __    _    ___ 
        |__E__|      _ __ ___    / \    \ \/ /   / \  |_ _|
     >===]__o[===<  | '_ ` _ \  / _ \    \  /   / _ \  | | 
         [o__]      | | | | | |/ ___ \   /  \  / ___ \ | | 
         /7 [|      |_| |_| |_/_/   \_\ /_/\_\/_/   \_\___| v.0
       \/7  [|_                                                
    ###########################################################

    Starting a XAI run in clean environment ...
    """)

    # Register the signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    model = PPO.load("/app/models/best_model.zip")
    env = gym.make("milliampereROS_env/MilliampereRos4Thrusters-v0", render_mode='human', max_time_steps=200)
    obs2action_model = Obs2ActionWrapper(model)
    obs2value_model = Obs2ValueWrapper(model)
    obs_dim = env.observation_space.shape[0]

    df = pd.read_csv("data.csv")
    runs_obs = df.iloc[:,:].values
    num_random = 500
    random_obs = np.array([env.observation_space.sample() for _ in range(num_random)])
    random_obs = np.concatenate([random_obs, -random_obs])
    #random_obs = np.zeros((1,14))

    #background_obs = np.concatenate((runs_obs[:501], random_obs))
    background_obs = random_obs
    background_torch = torch.tensor(background_obs, dtype=torch.float32)
    runs_torch = torch.tensor(runs_obs, dtype=torch.float32)

    explainer = shap.DeepExplainer(obs2action_model, background_torch)
    #explainer_value = shap.DeepExplainer(obs2value_model, background_torch)
    explainer_value = shap.DeepExplainer(obs2value_model, runs_torch)

    print(explainer.expected_value[:4]*1200)
    print(explainer.expected_value[4:]*180)

    base_vectors = [(explainer.expected_value[i]*1200, explainer.expected_value[i+4]*180) for i in range(4)]

    print(explainer_value.expected_value)
    render = RenderExplaination()
    fps = 30
    clock = pygame.time.Clock()
    sleep_time = 0.1
    shap_values_value_prev = np.zeros(14)
    explain_mode = 1

    agent = Agent()
    rospy.sleep(sleep_time*3)

    rospy.on_shutdown(shutdown_hook)
    try:
        while not rospy.is_shutdown():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("event quit")
                    rospy.signal_shutdown("User closed window")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    pause = True
                    print("Paused ...")
                    while pause:
                        time.sleep(0.1)
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                                pause = False
                                print("Unpaused ...")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    explain_mode *= -1
                    print("Switching explain mode") 

            
            # Test
            #x_tilde = 3 * math.cos(pygame.time.get_ticks() / 500)
            #y_tilde = 3 * math.sin(pygame.time.get_ticks() / 500)
            #psi_tilde = (pygame.time.get_ticks() / 10) % 360
            #vel_magnitude = 3.5
            # vel_angle = 180 * math.cos(pygame.time.get_ticks() / 800)
            # n_d = 1200/1200*3.5
            # alpha_d = -45
            # shap_value = 90
            # explain = Explain.VEL
            # x_tilde = 0
            # y_tilde = 0.5
            # psi_tilde = 0.25
            # u_hat = 0.02
            # v_hat = 0.02
            # vel_magnitude = 0
            # vel_angle = 0
            # n_d = 3.5*0.2
            # alpha_d = 90
            # shap_value = 90
            # explain = Explain.Y

            obs = agent.get_observations()

            while obs is None:
                print("waiting for observations")
                rospy.sleep(5)
                obs = agent.get_observations()

            vel_magnitude, vel_angle = combine_vectors([(obs.u_hat*3.5, 0), (obs.v_hat*3.5, math.pi/2)])
            #vel_magnitude, vel_angle = combine_vectors([(u_hat*3.5, 0), (v_hat*3.5, math.pi/2)])
            vel_angle = vel_angle/math.pi * 180

            actions = agent.get_actions()
            vectors = [
                (actions.n_d1, math.radians(actions.alpha_d1*180)),
                (actions.n_d2, math.radians(actions.alpha_d2*180)),
                (actions.n_d3, math.radians(actions.alpha_d3*180)),
                (actions.n_d4, math.radians(actions.alpha_d4*180)),
            ]
            n_d, alpha_d = combine_vectors(vectors)

            vectors = [
                (actions.n_d1, actions.alpha_d1*180),
                (actions.n_d2, actions.alpha_d2*180),
                (actions.n_d3, actions.alpha_d3*180),
                (actions.n_d4, actions.alpha_d4*180),
            ]
            alpha_d = alpha_d/math.pi *180
            #print(obs.x_tilde)
            
            obs_list = [
                obs.x_tilde,
                obs.y_tilde,
                obs.psi_tilde,
                obs.u_hat,
                obs.v_hat,
                obs.r_hat,
                obs.n_d1_prev,
                obs.n_d2_prev,
                obs.n_d3_prev,
                obs.n_d4_prev,
                obs.alpha_d1_prev,
                obs.alpha_d2_prev,
                obs.alpha_d3_prev,
                obs.alpha_d4_prev,
            ]

            obs_tensor = torch.tensor(obs_list, dtype=torch.float32).unsqueeze(0)
            shap_values_all = explainer.shap_values(obs_tensor)
            shap_values_value = explainer_value.shap_values(obs_tensor)
            #print(len(shap_values_value))
            # shap_values_all is a list of length=4 (one for each action dimension),
            # each is shape (1, 14) if you have 14 obs features.

            # # Extract each action’s SHAP array, shape => (14,)   # <<< fix
            # shap0 = shap_values_all[0][0]  # shape = (14,)
            # shap1 = shap_values_all[1][0]  # shape = (14,)
            # shap2 = shap_values_all[2][0]  # shape = (14,)
            # shap3 = shap_values_all[3][0]  # shape = (14,)
            # shap4 = shap_values_all[4][0]  # shape = (14,)
            # shap5 = shap_values_all[5][0]  # shape = (14,)
            # shap6 = shap_values_all[6][0]  # shape = (14,)
            # shap7 = shap_values_all[7][0]  # shape = (14,)
            
            # # If you want to “merge” shap2 and shap3:
            # shap2_3 = shap2 + shap3        # elementwise sum => shape=(14,)

            # Suppose you want to treat shap2_3 as if it replaced shap2 and shap3,
            # so effectively you have 3 "actions" for your analysis: shap0, shap1, and shap2_3.
            # Then sum their absolute values by feature:       # <<< fix
            #sum_of_abs_vals = np.abs(shap0) + np.abs(shap1) + np.abs(shap2) + np.abs(shap3) + np.abs(shap4) + np.abs(shap5) + np.abs(shap6) + np.abs(shap7)  # shape = (14,)
            # sum_of_abs_vals = np.abs(shap0) + np.abs(shap1) + np.abs(shap2) + np.abs(shap3)  # shape = (14,)
            # sum_of_abs_vals2 = np.abs(shap4) + np.abs(shap5) + np.abs(shap6) + np.abs(shap7)  # shape = (14,)
            #print(max(sum_of_abs_vals))

            # Find the feature index with largest overall importance:
            # max_index = np.argmax(sum_of_abs_vals)            # integer from 0..13
            # max_value = sum_of_abs_vals[max_index]
            # total_sum = sum_of_abs_vals.sum()
            # max_percent = 100.0 * max_value / total_sum

            #print(f"Feature index with highest importance: {max_index+1}, {max_percent:.2f}%")
            
            clock.tick(fps)

            # Then pass sum_of_abs_vals to your Pygame renderer:
            list_of_lists1 = [arr.flatten().tolist() for arr in shap_values_all[:4]]
            list_of_lists2 = [arr.flatten().tolist() for arr in shap_values_all[4:]]
            list_of_lists3 = shap_values_value - shap_values_value_prev
            try:
                #print("\n",shap_values_value)
                render.render_frame(
                    list_of_lists1,
                    list_of_lists2,
                    list_of_lists3.ravel()/10,
                    vectors,
                    n_d,
                    alpha_d,
                    obs.u_hat * 3.5 * 2,
                    obs.v_hat * 3.5 * 2,
                    360 * obs.r_hat * 2,
                    obs.y_tilde*10,
                    obs.x_tilde*10,
                    obs.psi_tilde*180,
                    explain_mode,
                    base_vectors
                )
                #shap_values_value_prev = shap_values_value.copy()
            except pygame.error as e:
                # This catches exceptions like "display Surface quit"
                print("Pygame error during rendering:", e)
                break



            # obs_tensor = torch.tensor(obs_list, dtype=torch.float32).unsqueeze(0)
        
            # shap_values = explainer.shap_values(obs_tensor)
            # #print(len(shap_values))
            # #print(len(shap_values[0]))
            # #print(len(shap_values[0][0]))
            # # Merge the 3rd and 4th elements (index 2 and 3)
            
            # merged_element = [a + b for a, b in zip(shap_values[2], shap_values[3])]
            # # Create a new list without the original 3rd and 4th elements, adding the merged one instead
            # new_data = shap_values[:2] + [merged_element] + shap_values[4:]
            # summed_shap_values = [sum(abs(x) for x in shap_vals) for shap_vals in zip(*new_data)]
            # #print(summed_shap_values)
            
            # max_index = np.argmax(summed_shap_values)
            # max_value = summed_shap_values[0][max_index]
            # total_sum = sum(summed_shap_values[0])
            # max_value = 100*max_value/total_sum
            # print(f"{max_index+1}, {max_value:.2f} %")

            # # Cap the frame rate
            # clock.tick(fps)
            # #render.render_frame(y_tilde * 10, x_tilde * 10, psi_tilde * 180, vel_magnitude*10, vel_angle, n_d, alpha_d, shap_value, explain)
            # render.render_frame(obs.y_tilde * 10, obs.x_tilde * 10, obs.psi_tilde * 180, vel_magnitude*10, vel_angle, n_d, alpha_d, max_value, max_index+1, summed_shap_values)
            try:
                rospy.sleep(sleep_time)
            except rospy.ROSInterruptException:
                break

    except Exception as e:
        print(f"Exception in main loop: {e}")
    finally:
        # Ensure cleanup happens regardless of how we exit the loop
        print("Quitting pygame...")
        pygame.quit()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Exception occurred:", e)
        pygame.quit()
        sys.exit(1)