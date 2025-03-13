from render_explanation import RenderExplaination
from stable_baselines3 import PPO
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import shap
import math
import pygame

model = PPO.load("/app/models/best_model.zip")
env = gym.make("")

class Obs2ActionWrapper(torch.nn.Module):
    def __init__(self, model):
        super(Obs2ActionWrapper, self).__init__()
        self.mlp_extractor = model.policy.mlp_extractor.policy_net
        self.action_net = model.policy.action_net
    
    def forward(self, obs):
        x = self.mlp_extractor(obs)
        action = self.action_net(x)
        return action

obs2action_model = Obs2ActionWrapper(model)

obs_dim = env.observation_space.shape[0]

df = pd.read_csv("")
runs_obs = df.iloc[:,:].values

num_random = 500
random_obs = np.array([env.observation_space.shape.sample() for _ in range(num_random)])

background_obs = np.concatenate((runs_obs[:501], random_obs))
background_torch = torch.tensor(background_obs, dtype=torch.float32)

explainer = shap.DeepExplainer(obs2action_model, background_torch)

def combine_vectors(vectors):
    total_x = sum(amplitude * math.cos(angle) for amplitude, angle in vectors)
    total_y = sum(amplitude * math.sin(angle) for amplitude, angle in vectors)

    resultant_magnitude = math.hypot(total_x, total_y)
    resultant_angle = math.atan2(total_y, total_x)
    
    return resultant_magnitude, resultant_angle

render = RenderExplaination()
fps = 30
clock = pygame.time.Clock()

runnig = True
while(runnig):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # EXPLAIN

    shap_values = explainer.shap_values(test_torch)
    act1,act2,act3,act4,act5,act6,act7,act8 = 0
    
    vectors = [
        (act1, math.radians(act5)),
        (act2, math.radians(act6)),
        (act3, math.radians(act7)),
        (act4, math.radians(act8)),
    ]
    
    thrust, angle = combine_vectors(vectors)
    vel_magnitude, vel_angle = combine_vectors([(u_hat, 0), (v_hat, math.pi/2)])

    # RENDER

    # Cap the frame rate
    clock.tick(fps)
    render.render_frame(x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, shap_value, explain)
    
pygame.quit()


