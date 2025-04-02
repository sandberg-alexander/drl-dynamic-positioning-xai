#!/usr/bin/env python3

# Fix the import - use the correct capitalization
import milliAmpere1ROS_env
from stable_baselines3 import PPO
import gymnasium as gym
from datetime import datetime
import traceback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

# Create a custom callback to save models after policy updates
class SaveModelCallback(BaseCallback):
    def __init__(self, save_interval=1, save_path=None, verbose=1):
        """
        Parameters:
        -----------
        save_interval : int
            Save every N policy updates (1 = save after every update)
        save_path : str
            Directory to save models
        verbose : int
            Verbosity level
        """
        super(SaveModelCallback, self).__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path
        self.policy_update_count = 0
        
    def _on_rollout_end(self):
        """This method is called after collecting rollout data but before updating the policy"""
        self.policy_update_count += 1
        
        if self.policy_update_count % self.save_interval == 0:
            # Save the model with timestamp and update count
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{self.save_path}/PPO_{timestamp}_steps_{self.num_timesteps}_update_{self.policy_update_count}"
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Saved model at {model_path}")
                
                # Log the current average reward based on recent episodes
                if len(self.model.ep_info_buffer) > 0:
                    mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                    print(f"Current mean reward: {mean_reward:.2f}")
        
        return True
    
    def _on_step(self):
        """Called at every step"""
        return True

def main():
    print("""
    ### ___T_ ################################################
       | n n |                   _      ____  ____  _         
       |__E__|      _ __ ___    / \    |  _ \|  _ \| |        
    >===]__o[===<  | '_ ` _ \  / _ \   | | | | |_) | |        
        [o__]      | | | | | |/ ___ \  | |_| |  _ <| |___     
        /7 [|      |_| |_| |_/_/   \_\ |____/|_| \_\_____| v.1
      \/7  [|_     train.py                                           
    ##########################################################

    Starting DRL training in clean environment ...
    """)
    
    # Create a timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"/app/models/training_{timestamp}"
    models_dir = f"{run_dir}/models"
    logs_dir = f"{run_dir}/logs"
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"Training run directory created at: {run_dir}")
    
    # Create and monitor the environment
    # Make sure the environment ID matches exactly what's registered
    env = gym.make("milliAmpere1ROS_env/MilliAmpere1ROS-v3", render_mode='human', max_time_steps=1000)
    
    # Monitor with logs in the run-specific directory
    env = Monitor(env, filename=f"{logs_dir}")
    
    # Setup model with explicitly defined hyperparameters
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=2048,
        verbose=1, 
        device='cuda'
    )
    
    # Setup callback for saving after policy updates
    # Save every 2 policy updates to the run-specific models directory
    save_callback = SaveModelCallback(save_interval=2, save_path=models_dir)
    
    try:
        # Start training without evaluation
        model.learn(total_timesteps=500000, callback=save_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving final model...")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Save final model to the run-specific models directory
        final_path = f"{models_dir}/FINAL_PPO_{timestamp}"
        model.save(final_path)
        print(f"Final model saved at {final_path}")
        env.close()

if __name__ == '__main__':
    main()