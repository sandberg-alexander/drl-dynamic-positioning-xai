################################################################################
## This script intends to train a Deep Reinforcment Learning (DRL) model
## with Stable-Baselines3 (https://stable-baselines3.readthedocs.io/en/master/), 
## using the custom made MilliampereROS-v1 enviorment from Gymnasium
## (https://gymnasium.farama.org/)
################################################################################
## Author: Alexander Sandberg
## Date: 2024-11-08
## Email: snadbeg@gmail.com
################################################################################

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

import gymnasium as gym
import numpy as np
import pygame
import milliampereROS_env
import rospy
from drl_msgs.msg import Observation, Action
import argparse

# def make_env(uri, hostname):
#     def _init():
#         import os
#         os.environ['ROS_MASTER_URI'] = uri
#         os.environ['ROS_HOSTNAME'] = hostname

#         # from milliampereROS_env.envs import MilliampereROSEnv
#         # return MilliampereROSEnv()
#         import importlib
#         env_mod = importlib.import_module("milliampereROS_env.envs")  
#         # This import happens *after* setting environment variables, in the sub-process
#         env_class = getattr(env_mod, "MilliampereROSEnv")
#         return env_class()
#     return _init

def main(args):
    print("""
    #### ___T_ ################################################
        | n n |                   _      ____  ____  _         
        |__E__|      _ __ ___    / \    |  _ \|  _ \| |        
     >===]__o[===<  | '_ ` _ \  / _ \   | | | | |_) | |        
         [o__]      | | | | | |/ ___ \  | |_| |  _ <| |___     
         /7 [|      |_| |_| |_/_/   \_\ |____/|_| \_\_____| v.0
       \/7  [|_                                                
    ###########################################################

    Starting a DRL run in clean environment ...
    """)

    # uris = os.environ['ROS_MASTER_URIS'].split(',')
    # hostnames = os.environ['ROS_HOSTNAMES'].split(',')

    # train_uris = [uri for uri in uris if "eval" not in uri]
    # train_hostnames = [hn for hn in hostnames if "eval" not in hn]
    # eval_uris = [uri for uri in uris if "eval" in uri]
    # eval_hostnames = [hn for hn in hostnames if "eval" in hn]

    # train_env = SubprocVecEnv([make_env(uri, hn) for (uri, hn) in zip(train_uris, train_hostnames)])
    # eval_env = SubprocVecEnv([make_env(uri, hn) for (uri, hn) in zip(eval_uris, eval_hostnames)])

    # model = PPO("MlpPolicy", train_env, verbose=1)

    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path="../models",
    #     log_path="../models/logs",
    #     eval_freq=1000,
    #     render=False,
    # )

    # model.learn(total_timesteps=3000, callback=eval_callback, progress_bar=True)
    # train_env.close()
    # eval_env.close()
    
    model = PPO.load("/app/models/best_model.zip")
    env = gym.make("milliampereROS_env/MilliampereRos4Thrusters-v0", render_mode='human', max_time_steps=args.max_timesteps)
    model.set_env(env)

    #rospy.init_node('ros_vessel_env2', anonymous=True)
    pub_observation = rospy.Publisher('/drl/observation', Observation, queue_size=1)
    pub_action = rospy.Publisher('/drl/action', Action, queue_size=1)

    record = False
    data = np.zeros((1000,14))
    j = 0

    observation, info = env.reset()
    i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(f"Terminating...")
                env.close()
                rospy.signal_shutdown('Environment closed')
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                print(f"Terminating...")
                env.close()
                rospy.signal_shutdown('Environment closed')
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if not record:
                    print(f"Recording data...")
                    record = True

                else:
                    print(f"Stoped recording with {j} datapoints")
                    record = False
        
        action, _ = model.predict(observation, deterministic=True)
        
        pub_observation.publish(*observation)
        pub_action.publish(*action)

        if record:
            try:
                data[j,:] = observation.ravel()
                if j%10 == 0:
                    print(j)
                j+=1
                
            except:
                print(f"Dataset full.")
                record = False
                np.savetxt("real_data3.csv", data, delimiter=",", fmt="%f", header="x,y,phi,u,v,r,n1,n2,n3,n4,alpha1,alpha2,alpha3,alpha4", comments="")
        observation, reward, terminated, truncated, info = env.step(action)
        i+=1

        if terminated or truncated:
            observation, info = env.reset()
            #record = False

    rospy.signal_shutdown('Environment closed')

def get_args():
    parser = argparse.ArgumentParser(description="A python script that runs DRL agent")
    parser.add_argument('--max_timesteps', type=int, default=1_000, help='specify max_timesteps for run')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  args = get_args()
  main(args)