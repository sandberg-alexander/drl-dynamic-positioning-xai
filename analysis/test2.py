from render_explanation import RenderExplaination, Explain
import pygame
import math
import rospy

from custom_msgs.msg import Observation, Action


class Agent():
    def __init__(self):
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

def combine_vectors(vectors):
    total_x = sum(amplitude * math.cos(angle) for amplitude, angle in vectors)
    total_y = sum(amplitude * math.sin(angle) for amplitude, angle in vectors)

    resultant_magnitude = math.hypot(total_x, total_y)
    resultant_angle = math.atan2(total_y, total_x)
    
    return resultant_magnitude, resultant_angle


render = RenderExplaination()
fps = 30
clock = pygame.time.Clock()
sleep_time = 0.1

agent = Agent()

runnig = True
while(runnig):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Test
    x_tilde = 3 * math.cos(pygame.time.get_ticks() / 500)
    y_tilde = 3 * math.sin(pygame.time.get_ticks() / 500)
    psi_tilde = (pygame.time.get_ticks() / 10) % 360
    vel_magnitude = 3.5
    vel_angle = 180 * math.cos(pygame.time.get_ticks() / 800)
    n_d = 1200/1200*3.5
    alpha_d = -45
    shap_value = 90
    explain = Explain.VEL


    x_tilde, y_tilde, psi_tilde, u_hat, v_hat, _ = agent.get_observations()
    vel_magnitude, vel_angle = combine_vectors([(u_hat, 0), (v_hat, math.pi/2)])

    actions = agent.get_actions()
    vectors = [
        (actions[0], math.radians(actions[4])),
        (actions[1], math.radians(actions[5])),
        (actions[2], math.radians(actions[6])),
        (actions[3], math.radians(actions[7])),
    ]
    n_d, alpha_d = combine_vectors(vectors)


    # Cap the frame rate
    clock.tick(fps)
    render.render_frame(x_tilde, y_tilde, psi_tilde, vel_magnitude, vel_angle, n_d, alpha_d, shap_value, explain)
    rospy.sleep(sleep_time)

pygame.quit()


