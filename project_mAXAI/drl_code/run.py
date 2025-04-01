from stable_baselines3 import PPO
import milliAmpere1ROS_env
import gymnasium as gym
import pygame
from gymnasium.wrappers import RecordVideo
import numpy as np

def main():
    model = PPO.load("/app/models/training_20250328_131443/models/best_model.zip")
    env = gym.make("milliAmpere1ROS_env/MilliAmpere1ROS-v3", render_mode='human', max_time_steps=200)
    model.set_env(env)

    # model = PPO("MlpPolicy", env, verbose=1)

    observation, info = env.reset()
    print("Press 'q' to terminate:")
    array = np.zeros((2048*3,14))
    i = 0
    while i<2048*3-1:
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
        array[i,:] = observation.ravel()
        i+=1
        if terminated or truncated:
            observation, info = env.reset()
        # The environment's render method is called within env.step()
        # So no need to call it here unless you prefer
    np.savetxt("array.csv", array, delimiter=",", fmt="%f", header="x,theta,u,r,a", comments="")

if __name__ == '__main__':
    main()
