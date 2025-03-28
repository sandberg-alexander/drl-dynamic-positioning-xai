#!/bin/bash
# setup_and_train.sh - Install the environment and run training

# Navigate to gym environment directory
cd /app/gym_env/milliAmpere1ROS_env

# Display directory contents to verify
echo "Contents of milliAmpere1ROS_env directory:"
ls -la

# Show setup.py
echo -e "\nContents of setup.py:"
cat setup.py

# Install the environment in development mode
echo -e "\nInstalling milliAmpere1ROS_env..."
pip install -e .

# Verify installation
echo -e "\nVerifying installation:"
pip list | grep milli

# Check available gym environments
echo -e "\nListing available gym environments:"
python3 -c "import gymnasium as gym; print([env_spec.id for env_spec in gym.envs.registry.all() if 'milli' in env_spec.id.lower()])"

# Return to the training directory
cd /app/drl_code

# Now create the improved training script
cat > train_improved.py << 'EOL'
#!/usr/bin/env python3
import sys
import os

# Add gym_env to Python path if needed
gym_env_path = "/app/gym_env"
if gym_env_path not in sys.path:
    sys.path.append(gym_env_path)

# Try to import the environment
try:
    import milliAmpere1ROS_env
    print("Successfully imported milliAmpere1ROS_env")
except ImportError as e:
    print(f"Error importing milliAmpere1ROS_env: {e}")
    # Try alternative import
    try:
        from milliAmpere1ROS_env import envs
        print("Imported via alternative path")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")

from stable_baselines3 import PPO
import gymnasium as gym
from datetime import datetime
import traceback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Print all available environments for debugging
print("\nAll available environments:")
all_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
print(all_envs)

print("\nLooking for MilliAmpere-related environments:")
milli_envs = [env for env in all_envs if 'milli' in env.lower()]
print(milli_envs)

# Create a custom callback to save models after policy updates
class SaveModelCallback(BaseCallback):
    def __init__(self, save_interval=1, save_path="/app/models", verbose=1):
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
        os.makedirs(save_path, exist_ok=True)
        
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
      \/7  [|_                                                
    ##########################################################

    Starting DRL training in clean environment ...
    """)
    
    # Try to find the correct environment ID
    try:
        # First attempt with the exact name from your code
        env_id = "MilliAmpere1ROS-v1"
        env = gym.make(env_id, render_mode='human', max_time_steps=3000)
        print(f"Successfully created environment: {env_id}")
    except gym.error.NameNotFound:
        # If that fails, try to find environment IDs containing 'milli'
        all_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
        milli_envs = [env for env in all_envs if 'milli' in env.lower()]
        
        if milli_envs:
            # Use the first matching environment
            env_id = milli_envs[0]
            print(f"Using alternative environment ID: {env_id}")
            env = gym.make(env_id, render_mode='human', max_time_steps=3000)
        else:
            print("No MilliAmpere environments found in the registry!")
            # Let's try to manually register the environment
            print("Attempting to manually register the environment...")
            try:
                from gymnasium.envs.registration import register
                register(
                    id="MilliAmpere1ROS-v1",
                    entry_point="milliAmpere1ROS_env.envs:MilliAmpere1ROSEnvV1",
                )
                env_id = "MilliAmpere1ROS-v1"
                env = gym.make(env_id, render_mode='human', max_time_steps=3000)
                print(f"Successfully registered and created environment: {env_id}")
            except Exception as e:
                print(f"Failed to manually register environment: {e}")
                return
    
    log_dir = "/app/models/training_logs/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=f"{log_dir}/ppo_{env_id}")
    
    # Setup model with explicitly defined hyperparameters
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=2048,
        verbose=1, 
        device='cuda'
    )
    
    # Setup callback for saving after policy updates
    # Save every 2 policy updates
    save_callback = SaveModelCallback(save_interval=2, save_path="/app/models")
    
    try:
        # Start training without evaluation
        model.learn(total_timesteps=500000, callback=save_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving final model...")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Save final model and close environment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = f"/app/models/FINAL_PPO_{timestamp}"
        model.save(final_path)
        print(f"Final model saved at {final_path}")
        env.close()

if __name__ == '__main__':
    main()
EOL

# Make the script executable
chmod +x train_improved.py

echo -e "\nSetup complete! You can now run the training with:"
echo "python3 train_improved.py"