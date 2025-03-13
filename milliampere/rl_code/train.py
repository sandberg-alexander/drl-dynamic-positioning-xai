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


def make_env(uri, hostname):
    def _init():
        import os
        os.environ['ROS_MASTER_URI'] = uri
        os.environ['ROS_HOSTNAME'] = hostname

        # from milliampereROS_env.envs import MilliampereROSEnv
        # return MilliampereROSEnv()
        import importlib
        env_mod = importlib.import_module("milliampereROS_env.envs")  
        # This import happens *after* setting environment variables, in the sub-process
        env_class = getattr(env_mod, "MilliampereROSEnv")
        return env_class()
    return _init

def main():
  print("""
  ### ___T_ ################################################
     | n n |                   _      ____  ____  _         
     |__E__|      _ __ ___    / \    |  _ \|  _ \| |        
  >===]__o[===<  | '_ ` _ \  / _ \   | | | | |_) | |        
      [o__]      | | | | | |/ ___ \  | |_| |  _ <| |___     
      /7 [|      |_| |_| |_/_/   \_\ |____/|_| \_\_____| v.0
    \/7  [|_                                                
  ##########################################################

  Starting DRL training in clean environment ...
  """)

  uris = os.environ['ROS_MASTER_URIS'].split(',')
  hostnames = os.environ['ROS_HOSTNAMES'].split(',')

  train_uris = [uri for uri in uris if "eval" not in uri]
  train_hostnames = [hn for hn in hostnames if "eval" not in hn]
  eval_uris = [uri for uri in uris if "eval" in uri]
  eval_hostnames = [hn for hn in hostnames if "eval" in hn]

  train_env = SubprocVecEnv([make_env(uri, hn) for (uri, hn) in zip(train_uris, train_hostnames)])
  eval_env = SubprocVecEnv([make_env(uri, hn) for (uri, hn) in zip(eval_uris, eval_hostnames)])

  model = PPO("MlpPolicy", train_env, verbose=1)

  eval_callback = EvalCallback(
      eval_env,
      best_model_save_path="../models",
      log_path="../models/logs",
      eval_freq=1000,
      render=False,
  )

  model.learn(total_timesteps=3000, callback=eval_callback, progress_bar=True)
  train_env.close()
  eval_env.close()


if __name__ == '__main__':
  main()