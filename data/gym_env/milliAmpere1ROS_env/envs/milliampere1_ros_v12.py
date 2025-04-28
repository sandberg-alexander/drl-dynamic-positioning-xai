import os
os.environ['SDL_VIDEO_X11_NO_MITSHM'] = '1'


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
from scipy.spatial.transform import Rotation
import pygame
import math

from geometry_msgs.msg import PoseStamped
from custom_msgs.msg import ActuatorSetpoints

class MilliAmpere1RosEnvV12(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, max_time_steps=3000, render_mode=None):

        low_action = np.array([-1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0])
        high_action = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=float)

        low_obs = np.concatenate([
            np.full(6, -1.0),  # First 6 elements with low bound -1
            low_action         # Last 8 elements with same low bounds as action space
        ])

        high_obs = np.concatenate([
            np.full(6, 1.0),   # First 6 elements with high bound 1
            high_action         # Last 8 elements with same high bounds as action space
        ])

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=float)

        # ROS publishers and subscribers
        rospy.init_node('ros_vessel_env', anonymous=True)
        self.pub_act_ref_1 = rospy.Publisher('/actuator_ref_1', ActuatorSetpoints, queue_size=1)
        self.pub_act_ref_2 = rospy.Publisher('/actuator_ref_2', ActuatorSetpoints, queue_size=1)
        self.pub_act_ref_3 = rospy.Publisher('/actuator_ref_3', ActuatorSetpoints, queue_size=1)
        self.pub_act_ref_4 = rospy.Publisher('/actuator_ref_4', ActuatorSetpoints, queue_size=1)
        rospy.Subscriber('/navigation/pose', PoseStamped, self._pose_callback)

        # ROS shutdown
        rospy.on_shutdown(self.close)

        # Constraints
        self.max_thruster_rpm = 900                 # RPM
        self.max_azimuth_angle = 90                 # degrees
        self.target_bounds = np.array([5,5,180])    # max meters from operating point in NED and max degrees
        self.max_distance = 10                      # max distance from target
        self.max_heading_angle = 180                # degrees
        self.max_linear_speed = 3.24                # m/s
        self.max_angular_speed = 112.6              # degrees/s
        self.max_time_steps = max_time_steps
        # Define angle constraints in radians
        self.actuator_constraints = [
            (np.pi/2, np.pi),       # Thruster 1: [90°, 180°]
            (-np.pi, -np.pi/2),     # Thruster 2: [-180°, -90°]
            (-np.pi/2, 0),          # Thruster 3: [-90°, 0°]
            (0, np.pi/2)            # Thruster 4: [0°, 90°]
        ]
        # Define opposite quadrants for each thruster
        self.opposite_quadrants = [
            (-np.pi/2, 0),          # Opposite of Thruster 1: [-90°, 0°]
            (0, np.pi/2),           # Opposite of Thruster 2: [0°, 90°]
            (np.pi/2, np.pi),       # Opposite of Thruster 3: [90°, 180°]
            (-np.pi, -np.pi/2)      # Opposite of Thruster 4: [-180°, -90°]
        ]

        # Enviorment variables
        self.navigation_pose_data = None
        self.act_ref = [ActuatorSetpoints() for _ in range(4)]
        self.sleep_time = 0.1
        self.episode_counter = 0
        self.north = 334.61                 # offset from NED origo in meters
        self.east = 990.24                  # offset from NED origo in meters

        self.obs_time = None
        self.obs_time_prev = None
        self.eta_obs = np.zeros(3)
        #self.eta_obs = np.array([self.north, self.east, 0])
        self.eta_obs_prev = self.eta_obs.copy()
        self.epsilon_obs = None
        self.epsilon = np.zeros(3)
        self.est_nu_obs = None
        self.est_nu = np.zeros(3)
        self.observation = None
        self.norm_observation = None
        self.action = np.zeros(8)
        self.action_prev = np.zeros(8)
        self.norm_action = np.zeros(8)
        self.norm_action_prev = np.zeros(8)
        self.real_azimuth_angle = 0.0
        self.thrusters = np.zeros(4)
        self.thrusters_prev = np.zeros(4)
        self.angles = np.array([135, -135, -45, 45])
        self.angles_prev = np.array([135, -135, -45, 45])
        self.actual_angles = np.zeros(4)

        # Reward weights and variance
        self.w_gauss            = 1.0
        self.w_AS_gauss         = 0.4
        self.w_const            = 0.3
        self.w_u                = 0.5
        self.w_v                = 0.5
        self.w_r                = 1.0
        self.w_abs_n            = 0.3
        self.w_est_dot_n        = 0.05
        self.w_est_dot_alpha    = 0.01
        self.w_constraint       = 0.15/(np.pi/4)
        # sigma values are squared
        sigma_d                 = 1.0       
        sigma_psi               = 5.0*50
        sigma_AS_d              = 5.0*5
        sigma_AS_psi            = 70.0*50
        simga = np.diag([sigma_d, sigma_psi])
        simga_AS = np.diag([sigma_AS_d, sigma_AS_psi])
        self.inv_sigma = np.linalg.inv(simga)
        self.inv_sigma_AS = np.linalg.inv(simga_AS)

        # Render init
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            self.width_meters, self.height_meters = 20, 20
            self.resolution = 50  # Pixels per meter
            self.width = self.width_meters * self.resolution
            self.height = self.height_meters * self.resolution
            self.bar_width = 30  # Width of the value bar
            self.bar_height = self.height - 30
            simga_render = np.diag([sigma_d,sigma_d])
            simga_AS_render = np.diag([sigma_AS_d,sigma_AS_d])
            self.inv_sigma_render = np.linalg.inv(simga_render)
            self.inv_sigma_AS_render = np.linalg.inv(simga_AS_render)
            x, y = np.meshgrid(
            np.linspace(0, self.width_meters, self.width),
            np.linspace(0, self.height_meters, self.height)
            )
            pos = np.dstack((x, y))
            diff = pos - np.array([10, 10])  # Centered around mean
            self.R_gauss_render = np.exp(-0.5 * np.einsum('...i,ij,...j', diff, self.inv_sigma_render, diff))
            self.R_AS_gauss_render = np.exp(-0.5 * np.einsum('...i,ij,...j', diff, self.inv_sigma_AS_render, diff))
            self.max_reward = 1.0 #self.w_gauss + self.w_AS_gauss
            self.min_reward = -0.1-0.1-0.1-1.0
                                  #np.exp(-0.5 * d**2 * self.inv_sigma[0][0])
                                  #np.exp(-0.5 * d**2 * self.inv_sigma_AS[0][0])

            self.window = pygame.display.set_mode((self.width + self.bar_width+50, self.height))
            pygame.display.set_caption('2D Multivariate Gaussian with Value Bar')
            self.clock = pygame.time.Clock()
    
    ######################################
    ######## callback functions ##########

    def _pose_callback(self, data):
        self.navigation_pose_data = data

    ######################################
    ########## main functions ############

    def reset(self, options=None, seed=None):
        super().reset(seed=seed)

        self.episode_counter += 1
        print(f"Episode: {self.episode_counter}")

        # Init episode variables
        self.time_step = 0
        self.terminated_flag = False
        self.target_pose = self.np_random.uniform(-1,1,3) * self.target_bounds + np.array([self.eta_obs[0], self.eta_obs[1], 0])
        #self.target_pose = self.np_random.uniform(-1,1,3) * self.target_bounds
        print(f"New target at {self.target_pose}")

        # Init actuators
        self._calculate_actuator_inputs(self.norm_action)
        self._pub_actuator_inputs()

        observation = self._get_obs()
        normalized_observation = self._get_norm_obs(observation)
        info = self._get_info()

        return normalized_observation, info



    def step(self, action):
        # Updating variables after new timestep
        self.time_step += 1
        self.eta_obs_prev = self.eta_obs.copy()
        self.action_prev = self.action.copy()
        self.norm_action_prev = self.norm_action.copy()
        self.obs_time_prev = self.obs_time
        self.thrusters_prev = self.thrusters.copy()
        self.angles_prev = self.angles.copy()

        # Doing new action
        self.norm_action = action.copy()
        self.action = action.copy()*self.max_thruster_rpm
        self._calculate_actuator_inputs(action)
        self._pub_actuator_inputs()

        # Sleep to collect observations from ROS node
        rospy.sleep(self.sleep_time)                    ##### SHOULD I REMOVE?

        observation = self._get_obs()
        normalized_observation = self._get_norm_obs(observation)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        reward = self._get_reward()
        info = self._get_info()
        self.render()

        return normalized_observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Calculate the R function
        z = self._calculate_reward()
        # No need to normalize z here unless required

        num_levels = 10

        # Target parameters
        orientation = self.target_pose[2]  # Update with actual orientation
        rect_width, rect_height = 2.86 * self.resolution, 5.06 * self.resolution

        # Additional rectangle parameters
        additional_orientation = self.eta_obs[2]  # Update with actual orientation

        additional_position = np.array([
            int(self.width // 2) - (self.epsilon_obs[1]) * self.resolution,
            int(self.height // 2) + (self.epsilon_obs[0]) * self.resolution
        ])

        # Clear the screen
        self.window.fill((255, 255, 255))

        # Draw the heatmap
        self._draw_heatmap(self.window, z, num_levels)

        # Draw the target
        self._draw_target(self.window, orientation, rect_width, rect_height)

        # Draw the additional rectangle
        self._draw_agent(self.window, additional_position, additional_orientation, rect_width, rect_height)

        pygame.draw.circle(self.window, (255, 0, 0), (self.width // 2, self.height // 2), int(10.0 * self.resolution), 2)

        # Interpolate z_value at floating-point position
        from scipy.ndimage import map_coordinates
        z_value = np.clip(map_coordinates(z, [[additional_position[1]], [additional_position[0]]], order=1)[0],self.min_reward, self.max_reward)

        # Determine color based on z_value
        if z_value >= 0:
            intensity = int((z_value / self.max_reward) * 255)
            center_color =  np.clip((255 - intensity, 255, 255 - intensity),0,255)  # Transition from white to dark green
        else:
            intensity = int((abs(z_value) / abs(self.max_reward)) * 255)
            center_color = np.clip((255, 255 - intensity, 255 - intensity),0,255)  # Transition from white to dark red
        
        # Draw the value bar
        self._draw_value_bar(self.window, z_value*10, center_color)

        # Draw additional value bars for negative rewards
        self._draw_horizontal_value_bars(self.window, [self.R_vel, self.R_thrust, self.R_thrust_d, self.R_angle_d], 4)

        # Draw observations on the screen
        self._draw_observations(self.window)

        # Update the display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.render_mode == "human":
            pygame.quit()
        rospy.signal_shutdown('Environment closed')

    
    ######################################
    ######### helper functions ###########

    def _get_obs(self):

        # Makes sure to get navigation_pose_data before prociding
        while self.navigation_pose_data is None:
            rospy.sleep(self.sleep_time)

        # Retriving agents pose from navigation_pose_data
        self.obs_time = self.navigation_pose_data.header.stamp
        self.eta_obs[0] = self.navigation_pose_data.pose.position.x     # x NED-frame
        self.eta_obs[1] = self.navigation_pose_data.pose.position.y     # y NED-frame
        quarternian = self.navigation_pose_data.pose.orientation        
        quarternian = [quarternian.x, quarternian.y, quarternian.z, quarternian.w] 
        rotation = Rotation.from_quat(quarternian)                      
        self.eta_obs[2] = rotation.as_euler('zxy', degrees = True)[0]   # psi (heading)

        # Calculating epsilon (error from current pose to target pose)
        self.epsilon_obs = self.target_pose - self.eta_obs
        self.epsilon_obs[2] = self._ssa(self.target_pose[2]-self.eta_obs[2])
        epsilon_x_b, epsilon_y_b = self._rotate_ned2body() @ self.epsilon_obs[:2]
        self.epsilon = np.array([epsilon_x_b, epsilon_y_b, self.epsilon_obs[2]])

        # Calculating velocity estimates
        if self.obs_time_prev is not None:
            # Retriving observation time
            delta_t = (self.obs_time - self.obs_time_prev).to_sec()
            
            self.est_nu_obs = (self.eta_obs - self.eta_obs_prev)/delta_t
            self.est_nu_obs[2] = self._ssa(self.eta_obs[2]-self.eta_obs_prev[2])/delta_t     
            est_u, est_v = self._rotate_ned2body() @ self.est_nu_obs[:2]   # [u,v,r]
            self.est_nu = np.array([est_u, est_v, self.est_nu_obs[2]])

        # Agents observation at timestep t
        self.observation = np.concatenate((self.epsilon, self.est_nu, self.action_prev))

        return self.observation

    def _get_norm_obs(self, observation):
        epsilon_x_b = observation[0]
        epsilon_y_b = observation[1]
        epsilon_psi = observation[2]
        est_u = observation[3]
        est_v = observation[4]
        est_r = observation[5]

        norm_epsilon_x_b = min(1, max(-1, epsilon_x_b / self.max_distance))
        norm_epsilon_y_b = min(1, max(-1, epsilon_y_b / self.max_distance))
        norm_epsilon_psi = epsilon_psi / self.max_heading_angle
        norm_est_u = min(1, max(-1, est_u / self.max_linear_speed))
        norm_est_v = min(1, max(-1, est_v / self.max_linear_speed))
        norm_est_r = min(1, max(-1, est_r / self.max_angular_speed))

        norm_list = np.array([norm_epsilon_x_b, norm_epsilon_y_b, norm_epsilon_psi, norm_est_u, norm_est_v, norm_est_r])
        self.norm_observation = np.concatenate((norm_list, self.norm_action_prev))
        
        return self.norm_observation

    def _get_info(self):
        return {
            "observation": self.observation,
            "target_pose": self.target_pose,
            "thrusters": self.thrusters_prev,
            "angles": self.angles_prev,
            "actual_angles": self.actual_angles*180/np.pi
        }       ###### PUT MORE INFO!!!
    
    def _is_terminated(self):
        epsilon_x_b = self.epsilon[0]
        epsilon_y_b = self.epsilon[1]
        epsilon_d = np.sqrt(epsilon_x_b**2+epsilon_y_b**2)
        if epsilon_d > 10:
            print(f"Episode ended due to t-ERM-ination at epsilon_d = {epsilon_d}")
            self.terminated_flag = True
            return True

        else:
            return False
        

    def _is_truncated(self):
        if self.time_step > self.max_time_steps:
            print(f"Episode ended due to t-RUNC-ation at epsilon = {self.epsilon}")
            return True
        else: return False
    
    def _get_reward(self):
        d = np.sqrt(self.epsilon[0]**2 + self.epsilon[1]**2)
        epsilon_psi = self.epsilon[2]
        norm_est_u, norm_est_v, est_r = self.norm_observation[3:6]
        x = np.array([d, abs(epsilon_psi)]).T

        self.R_gauss = self.w_gauss * np.exp(-0.5 * x.T @ self.inv_sigma @ x)
        self.R_gauss_psi = self.R_gauss / np.exp(-0.5 * d**2 * self.inv_sigma[0][0])
        
        self.R_AS_gauss = self.w_AS_gauss * np.exp(-0.5 * x.T @ self.inv_sigma_AS @ x)
        self.R_AS_gauss_psi = self.R_AS_gauss / np.exp(-0.5 * d**2 * self.inv_sigma_AS[0][0])
        
        #self.R_vel = - 0.1/np.sqrt(3) * np.sqrt(norm_est_u**2 + norm_est_v**2 + est_r**2)
        b=1000
        x= np.sqrt(norm_est_u**2 + norm_est_v**2 + est_r**2)
        self.R_vel = -0.1*(b**(-x)-1)/(b**(-np.sqrt(3))-1)
        self.R_thrust = -0.1/4*(abs(self.thrusters[0])+abs(self.thrusters[1])+abs(self.thrusters[2])+abs(self.thrusters[3]))/self.max_thruster_rpm
        self.R_thrust_d = -0.1/8*(abs(self.thrusters[0]-self.thrusters_prev[0])+abs(self.thrusters[1]-self.thrusters_prev[1])+abs(self.thrusters[2]-self.thrusters_prev[2])+abs(self.thrusters[3]-self.thrusters_prev[3]))/self.max_thruster_rpm
        self.R_angle_d = -1.0/4*(abs(self.angles[0]-self.angles_prev[0])+abs(self.angles[1]-self.angles_prev[1])+abs(self.angles[2]-self.angles_prev[2])+abs(self.angles[3]-self.angles_prev[3]))/90

        self.R_rest = self.R_vel + self.R_thrust + self.R_thrust_d + self.R_angle_d
        # R_act = 0 
        # for i in range(4):
        #     n = self.norm_action[i]
        #     n_prev = self.norm_action_prev[i]
        #     alpha = self.norm_action[i+4]
        #     alpha_prev = self.norm_action_prev[i+4]
        #     est_dot_n = (n-n_prev)/self.delta_t_action
        #     est_dot_alpha = (alpha-alpha_prev)/self.delta_t_action
            
        #     R_act -= self.w_abs_n * abs(n) - self.w_est_dot_n * abs(est_dot_n) - self.w_est_dot_alpha * abs(est_dot_alpha)
        if self.terminated_flag:
            self.R_termination = -100
        else:
            self.R_termination = 0
            
        reward = np.squeeze(self.R_gauss + self.R_AS_gauss)/1.4 + self.R_vel + self.R_thrust + self.R_thrust_d + self.R_angle_d + self.R_termination# + R_act
        
        return reward

    def _calculate_actuator_inputs(self, action):

        # Process each thruster
        for i in range(4):
            x = action[i*2]
            y = action[i*2 + 1]

            if x == 0 and y == 0:
                # add 0 thrust and angle
                self.thrusters[i] = 0
                self.angles[i] = self.angles_prev[i]
                continue

            angle = np.arctan2(y, x)
            if angle == np.pi and i ==1:
                angle = -np.pi
            self.actual_angles[i] = angle
            thrust = np.clip(np.sqrt(x**2 + y**2), 0, 1)
            
            self.angles[i] = angle / np.pi * 180
            self.thrusters[i] = thrust * self.max_thruster_rpm


    def _pub_actuator_inputs(self):
        
        for i in range(len(self.act_ref)):          
            self.act_ref[i].throttle_reference = round(self.thrusters[i])
            self.act_ref[i].angle_reference = round(self.angles[i])

        self.pub_act_ref_1.publish(self.act_ref[0])
        self.pub_act_ref_2.publish(self.act_ref[1])
        self.pub_act_ref_3.publish(self.act_ref[2])
        self.pub_act_ref_4.publish(self.act_ref[3])

    def _rotate_ned2body(self):
        psi = np.deg2rad(self.eta_obs[2])
        c = np.cos(psi)
        s = np.sin(psi)
        return np.array([[c,s],[-s,c]])

    def _ssa(self, angle):
        return (angle+180) % 360 - 180
    
    def _ssa_rad(self, angle):
        return (angle+np.pi) % (2*np.pi) - np.pi
    
    def _ssa_alt(self, angle):
        if angle == -np.pi:
            return np.pi
        return (angle+np.pi) % (2*np.pi) - np.pi
    
    ######################################
    ######### render functions ###########

    def _calculate_reward(self):
        R_gauss = self.R_gauss_render * self.R_gauss_psi
        R_AS_gauss = self.R_AS_gauss_render * self.R_AS_gauss_psi
        return np.squeeze(R_gauss + R_AS_gauss)/1.4 + self.R_rest
    
    def _draw_heatmap(self, screen, z, num_levels):
        min_value = round(self.min_reward*10)
        max_value = round(self.max_reward*10)
        p_num_levels = abs(max_value*2)
        n_num_levels = abs(min_value*2)

        color_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(max(p_num_levels,n_num_levels) + 1):
            pos_threshold = (i / p_num_levels) * max_value
            next_pos_threshold = ((i + 1) / p_num_levels) * max_value
            neg_threshold = -(i / n_num_levels) * abs(min_value)
            next_neg_threshold = -((i + 1) / n_num_levels) * abs(min_value)
            mask_pos = (10*z >= pos_threshold) & (10*z < next_pos_threshold)
            mask_neg = (10*z < neg_threshold) & (10*z >= next_neg_threshold)
            p_intensity = int((i/ (p_num_levels)) * 255)
            n_intensity = int((i/ (p_num_levels)) * 255)
            
            # Set negative values to transition from white to dark red
            color_map[..., 0][mask_neg] = 255  # Red channel for negative values
            color_map[..., 1][mask_neg] = 255 - n_intensity  # Green channel
            color_map[..., 2][mask_neg] = 255 - n_intensity  # Blue channel
            # Set positive values to transition from white to dark green
            color_map[..., 0][mask_pos] = 255 - p_intensity  # Red channel
            color_map[..., 1][mask_pos] = 255              # Green channel
            color_map[..., 2][mask_pos] = 255 - p_intensity  # Blue channel
        surface = pygame.surfarray.make_surface(color_map)
        screen.blit(surface, (0, 0))
    
    def _draw_dashed_line(self, surface, color, start_pos, end_pos, dash_length=10, width=1):
        x1, y1 = start_pos
        x2, y2 = end_pos

        # Calculate the total length of the line
        total_length = math.hypot(x2 - x1, y2 - y1)
        if total_length == 0:
            return

        # Calculate the number of dashes
        dash_count = max(int(total_length / dash_length), 1)

        # Calculate the x and y increments
        x_increment = (x2 - x1) / dash_count
        y_increment = (y2 - y1) / dash_count

        for i in range(dash_count):
            if i % 2 == 0:
                start = (x1 + x_increment * i, y1 + y_increment * i)
                end = (x1 + x_increment * (i + 1), y1 + y_increment * (i + 1))
                pygame.draw.line(
                    surface, color,
                    (round(start[0]), round(start[1])),
                    (round(end[0]), round(end[1])),
                    width
                )
    
    def _draw_dashed_rect(self, surface, color, rect, dash_length=10, width=1):
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        w -= 2
        h -= 2
        # Top edge
        self._draw_dashed_line(surface, color, (x, y), (x + w, y), dash_length, width)
        # Right edge
        self._draw_dashed_line(surface, color, (x + w, y), (x + w, y + h), dash_length, width)
        # Bottom edge
        self._draw_dashed_line(surface, color, (x + w, y + h), (x, y + h), dash_length, width)
        # Left edge
        self._draw_dashed_line(surface, color, (x, y + h), (x, y), dash_length, width)
    
    def _draw_target(self, screen, orientation, rect_width, rect_height):
        rect_center = (self.width // 2, self.height // 2)
        target_surface = pygame.Surface((int(rect_width), int(rect_height)), pygame.SRCALPHA)
        target_rect = target_surface.get_rect()
        target_rect.center = (rect_width / 2, rect_height / 2)

        # Draw dashed rectangle outline
        self._draw_dashed_rect(
            target_surface,
            color=(128, 128, 128),  # Gray color
            rect=target_surface.get_rect(),
            dash_length=10,
            width=2
        )

        # Draw arrow indicating the front
        arrow_tip = (rect_width / 2, 10)
        arrow_left = (rect_width / 2 - 10, 30)
        arrow_right = (rect_width / 2 + 10, 30)
        pygame.draw.polygon(
            target_surface,
            color=(128, 128, 128),  # Gray color
            points=[
                (round(arrow_tip[0]), round(arrow_tip[1])),
                (round(arrow_left[0]), round(arrow_left[1])),
                (round(arrow_right[0]), round(arrow_right[1]))
            ]
        )

        pygame.draw.circle(screen, (128, 128, 128), rect_center, 3)

        # Rotate the target surface
        rotated_surface = pygame.transform.rotate(target_surface, -orientation)
        new_rect = rotated_surface.get_rect(center=rect_center)

        # Blit the rotated surface onto the main screen
        screen.blit(rotated_surface, new_rect.topleft)

    def _draw_agent(self, screen, position, orientation, rect_width, rect_height):
        target_rect = pygame.Rect(0, 0, rect_width, rect_height)
        target_rect.center = position
        target_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
        target_surface.fill((0, 0, 255, 128))  # Blue color with transparency
        arrow_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
        pygame.draw.polygon(arrow_surface, (0, 0, 0), [(rect_width // 2, 0), (rect_width // 2 - 10, 20), (rect_width // 2 + 10, 20)])  # Arrow pointing up
        target_surface.blit(arrow_surface, (0, 0))
        rotated_surface = pygame.transform.rotate(target_surface, -orientation)
        new_rect = rotated_surface.get_rect(center=position)
        screen.blit(rotated_surface, new_rect.topleft)
        # Draw a small dot in the middle of the blue rectangle
        pygame.draw.circle(screen, (0, 0, 0), position, 3)


    def _draw_value_bar(self, screen, value, color):
        min_value = self.min_reward*10
        max_value = self.max_reward*10

        # Add border around the bar
        border_thickness = 4
        bar_x = self.width + border_thickness
        bar_y = (self.height - self.bar_height) // 2 + border_thickness
        bar_width = self.bar_width
        bar_height = self.bar_height

        # Draw border
        border_rect = pygame.Rect(
            self.width,
            (self.height - self.bar_height) // 2,
            self.bar_width + border_thickness * 2,
            self.bar_height + border_thickness * 2
        )
        pygame.draw.rect(screen, (0, 0, 0), border_rect)

        # Draw the bar background
        inner_rect = pygame.Rect(
            bar_x,
            bar_y,
            bar_width,
            bar_height
        )
        pygame.draw.rect(screen, (200, 200, 200), inner_rect)

        # Map the value to position on the bar
        def value_to_bar_position(val):
            return bar_y + (max_value - val) / (max_value - min_value) * bar_height

        # Draw the filled bar representing the current value
        zero_position = value_to_bar_position(0)
        value_position = value_to_bar_position(value)

        if value >= 0:
            bar_top = value_position
            bar_height_draw = zero_position - value_position
        else:
            bar_top = zero_position
            bar_height_draw = value_position - zero_position

        # Ensure the bar height is positive
        bar_height_draw = abs(bar_height_draw)

        pygame.draw.rect(
            screen,
            color,
            (bar_x, bar_top, bar_width, bar_height_draw)
        )

        # Draw markers and labels
        marker_interval = 1  # Adjust this to set the interval between markers
        marker_values = list(range(int(min_value), int(max_value) + 1, marker_interval))

        for val in marker_values:
            marker_position = value_to_bar_position(val)
            pygame.draw.line(
                screen, (0, 0, 0),
                (bar_x, marker_position),
                (bar_x + bar_width, marker_position), 2
            )
            font = pygame.font.SysFont(None, 24)
            if val >= 0:
                label = f'+{val/10.0}'
            else:
                label = f'{val/10.0}'  # Display integer values
            text = font.render(label, True, (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.midleft = (bar_x + bar_width +5, marker_position)
            screen.blit(text, text_rect)

    def _draw_horizontal_value_bars(self, screen, values, num_bars):
        min_value = -0.1 * 10
        max_value = 0 * 10
        bar_width = (self.bar_width - 10) // num_bars
        bar_spacing = 10  # Space between bars
        bar_x_offset = self.width - 70
        bar_height = 75  # Height of the bars
        min_value_alt = -1.0*10

        for i, value in enumerate(values):
            if i == 3:
                min_value = min_value_alt
            bar_x = bar_x_offset + i * (bar_width + bar_spacing)
            bar_y = self.height - bar_height - 10
            # Draw the bar background
            pygame.draw.rect(
                screen,
                (255, 0, 0),
                (bar_x, bar_y, bar_width, bar_height)
            )

            # Calculate and draw the filled portion of the bar
            fill_height = min(max(0, (value*10 - min_value) / (max_value - min_value) * bar_height), bar_height)
            fill_y = bar_y + bar_height - fill_height
            color = (200, 200, 200) if value <= 0 else (0, 255, 0)

            pygame.draw.rect(
                screen,
                color,
                (bar_x, fill_y, bar_width, fill_height)
            )

        # Draw markers and labels
        marker_values = list(range(int(min_value), int(max_value) + 1, 1))
        for val in marker_values:
            marker_position = bar_y + (max_value - val) / (max_value - min_value) * bar_height
            pygame.draw.line(
                screen, (0, 0, 0),
                (bar_x, marker_position),
                (bar_x + bar_width, marker_position), 1
            )
            font = pygame.font.SysFont(None, 18)
            label = f'{val / 10.0}'  # Display decimal values
            text = font.render(label, True, (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.midleft = (bar_x + bar_width + 5, marker_position)
            screen.blit(text, text_rect)
    
    def _draw_observations(self, screen):
        font = pygame.font.SysFont(None, 24)
        text_color = (0, 0, 0)
        observations = [
            f'epsilon_x_n = {self.epsilon_obs[0]:.3f}',
            f'epsilon_y_n = {self.epsilon_obs[1]:.3f}',
            f'epsilon_x_b = {self.observation[0]:.3f}',
            f'epsilon_y_b = {self.observation[1]:.3f}',
            f'epsilon_psi = {self.observation[2]:.3f}',
            f'est_u = {self.observation[3]:.3f}',
            f'est_v = {self.observation[4]:.3f}',
            f'est_r = {self.observation[5]:.3f}',
            f'prev_action = {self.observation[6:]}',
            f'n_epsilon_x_b = {self.norm_observation[0]:.3f}',
            f'n_epsilon_y_b = {self.norm_observation[1]:.3f}',
            f'n_epsilon_psi = {self.norm_observation[2]:.3f}',
            f'n_est_u = {self.norm_observation[3]:.3f}',
            f'n_est_v = {self.norm_observation[4]:.3f}',
            f'n_est_r = {self.norm_observation[5]:.3f}',
            f'n_prev_action = {self.norm_observation[6:]}',
            f'action = {self.action}',
            f'n_action = {self.norm_action}',
            f'thrusters = {self.thrusters}',
            f'thrusters_prev = {self.thrusters_prev}',
            f'angles = {self.angles}',
            f'angles_prev = {self.angles_prev}',
            f'time_step = {self.time_step}',
        ]

        x, y = 10, 10
        for obs in observations:
            text = font.render(obs, True, text_color)
            screen.blit(text, (x, y))
            y += 30
