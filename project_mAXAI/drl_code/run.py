from stable_baselines3 import PPO
import milliAmpere1ROS_env
import gymnasium as gym
import pygame
from gymnasium.wrappers import RecordVideo
import numpy as np
import rospy
from custom_ros_msgs.msg import ObservationActuatorRefPair

def main():
    print("""
    ### ___T_ ################################################
       | n n |                   _      ____  ____  _         
       |__E__|      _ __ ___    / \    |  _ \|  _ \| |        
    >===]__o[===<  | '_ ` _ \  / _ \   | | | | |_) | |        
        [o__]      | | | | | |/ ___ \  | |_| |  _ <| |___     
        /7 [|      |_| |_| |_/_/   \_\ |____/|_| \_\_____| v.1
      \/7  [|_     run.py                                           
    ##########################################################

    Starting DRL training in clean environment ...
    """)

    model = PPO.load("/app/models/training_20250328_131443/models/best_model.zip")
    env = gym.make("milliAmpere1ROS_env/MilliAmpere1ROS-v3", render_mode='human', max_time_steps=200)
    model.set_env(env)
    
    pub_obs_act_ref_pair = rospy.Publisher('/drl/observation_actuator_ref_pair', ObservationActuatorRefPair, queue_size=1)
    obs_act_ref_pair = ObservationActuatorRefPair()

    observation, info = env.reset()
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Terminating...")
                env.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                print("Terminating...")
                env.close()
                return

        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        obs_act_ref_pair.x_tilde = observation[0]
        obs_act_ref_pair.y_tilde = observation[1]
        obs_act_ref_pair.psi_tilde = observation[2]
        obs_act_ref_pair.u_hat = observation[3]
        obs_act_ref_pair.v_hat = observation[4]
        obs_act_ref_pair.r_hat = observation[5]
        obs_act_ref_pair.n_x1d_prev = observation[6]
        obs_act_ref_pair.n_y1d_prev = observation[7]
        obs_act_ref_pair.n_x2d_prev = observation[8]
        obs_act_ref_pair.n_y2d_prev = observation[9]
        obs_act_ref_pair.n_x3d_prev = observation[10]
        obs_act_ref_pair.n_y3d_prev = observation[11]
        obs_act_ref_pair.n_x4d_prev = observation[12]
        obs_act_ref_pair.n_y4d_prev = observation[13]
        obs_act_ref_pair.n_d1 = info['thrusters'][0]
        obs_act_ref_pair.alpha_d1 = info['angles'][0]
        obs_act_ref_pair.n_d2 = info['thrusters'][1]
        obs_act_ref_pair.alpha_d2 = info['angles'][1]
        obs_act_ref_pair.n_d3 = info['thrusters'][2]
        obs_act_ref_pair.alpha_d3 = info['angles'][2]
        obs_act_ref_pair.n_d4 = info['thrusters'][3]
        obs_act_ref_pair.alpha_d4 = info['angles'][3]

        pub_obs_act_ref_pair.publish(obs_act_ref_pair)


        if terminated or truncated:
            observation, info = env.reset()

if __name__ == '__main__':
    main()
