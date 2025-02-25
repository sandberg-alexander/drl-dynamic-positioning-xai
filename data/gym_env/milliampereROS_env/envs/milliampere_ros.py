import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial.transform import Rotation
import pygame
import rospy


from geometry_msgs.msg import PoseStamped
from custom_msgs.msg import ActuatorSetpoints
from sim_milliampere.srv import ResetState
from supervisor.srv import SwitchMode

class MilliampereROSEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, max_time_steps=100, render_mode=None):
        print(1)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=float)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=float)

        # ROS publishers and subscribers
        #import rospy
        rospy.init_node('ros_vessel_env', anonymous=True)
        self.pub_act_ref_1 = rospy.Publisher('/actuator_ref_1', ActuatorSetpoints, queue_size=1)
        self.pub_act_ref_2 = rospy.Publisher('/actuator_ref_2', ActuatorSetpoints, queue_size=1)
        self.pub_act_ref_3 = rospy.Publisher('/actuator_ref_3', ActuatorSetpoints, queue_size=1)
        self.pub_act_ref_4 = rospy.Publisher('/actuator_ref_4', ActuatorSetpoints, queue_size=1)
        rospy.Subscriber('/navigation/pose', PoseStamped, self._pose_callback)
        
        import os
        def get_ros_master_uri():
            # Retrieve ROS Master URI from the environment
            ros_master_uri = os.environ.get('ROS_MASTER_URI')
            if ros_master_uri:
                return ros_master_uri
            else:
                return "ROS_MASTER_URI is not set in the environment."
        print(get_ros_master_uri())
        print(rospy.get_namespace())

        # ROS services
        self.reset_service = rospy.ServiceProxy('/sim_vessel/reset_state', ResetState)
        self.switch_mode_servic = rospy.ServiceProxy('/supervisor/switch_mode', SwitchMode)
        # Switch to direct actuaor control
        rospy.wait_for_service('/supervisor/switch_mode')
        try:
            self.switch_mode_servic('direct_actuator_control')
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

        # ROS shutdown
        rospy.on_shutdown(self.close)

        # Constraints
        self.max_thruster_rpm = 1200                # RPM
        self.max_azimuth_angle = 180                # degrees
        self.target_bounds = np.array([5,5,180])    # max meters from NED origo and max degrees
        self.max_distance = 10                      # max distance from target
        self.max_heading_angle = 180                #
        self.max_linear_speed = 3.5
        self.max_angular_speed = 120                 ##### NEED TO TUNE!!!
        self.max_time_steps = max_time_steps

        # Enviorment variables
        self.navigation_pose_data = None
        self.act_ref = [ActuatorSetpoints() for _ in range(4)]
        self.sleep_time = 0.1
        self.episode_counter = 0

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
            self.min_reward = -0.1-0.1-0.1
                                  #np.exp(-0.5 * d**2 * self.inv_sigma[0][0])
                                  #np.exp(-0.5 * d**2 * self.inv_sigma_AS[0][0])

            self.window = pygame.display.set_mode((self.width + self.bar_width+50, self.height))
            pygame.display.set_caption('2D Multivariate Gaussian with Value Bar')
            self.clock = pygame.time.Clock()
        print(2)
    ######################################
    ########## init functions ############
    
    def _init_episode_var(self):
        self.time_step = 0
        self.target_pose = self.np_random.uniform(-1,1,3) * self.target_bounds
        self.obs_time = None
        self.obs_time_prev = None
        self.eta_obs = np.zeros(3)
        self.eta_obs_prev = np.zeros(3)
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
    
    ######################################
    ######## callback functions ##########

    def _pose_callback(self, data):
        #print(2.5)
        self.navigation_pose_data = data

    ######################################
    ########## main functions ############

    def reset(self, options=None, seed=None):
        print(3)
        super().reset(seed=seed)

        self.episode_counter += 1
        print(f"Episode: {self.episode_counter}")

        # Init episode variables
        self._init_episode_var()
        print(f"New target at {self.target_pose}")

        # Init actuators
        self._pub_actions(self.action)

        # Init episode enviorment
        rospy.wait_for_service('/sim_vessel/reset_state')
        try:
            self.reset_service(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

        # Sleep to collect observations from ROS node
        rospy.sleep(self.sleep_time*30)                    ##### SHOULD I REMOVE?

                # Init episode enviorment
        rospy.wait_for_service('/sim_vessel/reset_state')
        try:
            self.reset_service(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        
        rospy.sleep(self.sleep_time)

        observation = self._get_obs()
        normalized_observation = self._get_norm_obs(observation)
        info = self._get_info()
        print(4)
        return normalized_observation, info



    def step(self, action):
        print(5)
        # Updating variables after new timestep
        self.time_step += 1
        self.eta_obs_prev = self.eta_obs.copy()
        self.action_prev = self.action.copy()
        self.norm_action_prev = self.norm_action.copy()
        self.obs_time_prev = self.obs_time

        # Doing new action
        self.norm_action = action.copy()
        self._pub_actions(action)

        # Sleep to collect observations from ROS node
        rospy.sleep(self.sleep_time)                    ##### SHOULD I REMOVE?

        observation = self._get_obs()
        normalized_observation = self._get_norm_obs(observation)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        reward = self._get_reward()
        info = self._get_info()
        self.render()
        print(6)
        return normalized_observation, reward, terminated, truncated, info

    def render(self):
        print(7)
        return

    def close(self):
        print(8)
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
            "target_pose": self.target_pose
        }       ###### PUT MORE INFO!!!
    
    def _is_terminated(self):
        epsilon_x_b = self.epsilon[0]
        epsilon_y_b = self.epsilon[1]
        epsilon_d = np.sqrt(epsilon_x_b**2+epsilon_y_b**2)
        if epsilon_d > 10:
            print(f"Episode ended due to termination at epsilon_d = {epsilon_d}")
            return True
        
        else:
            return False
        

    def _is_truncated(self):
        if self.time_step > self.max_time_steps:
            print(f"Episode ended due to truncation at epsilon = {self.epsilon}")
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
        self.R_thrust = -0.1/4*(abs(self.norm_action[0])+abs(self.norm_action[1])+abs(self.norm_action[2])+abs(self.norm_action[3]))
        self.R_thrust_d = -0.1/8*(abs(self.norm_action[0]-self.norm_action_prev[0])+abs(self.norm_action[1]-self.norm_action_prev[1])+abs(self.norm_action[2]-self.norm_action_prev[2])+abs(self.norm_action[3]-self.norm_action_prev[3]))
        self.R_rest = self.R_vel + self.R_thrust + self.R_thrust_d
        # R_act = 0 
        # for i in range(4):
        #     n = self.norm_action[i]
        #     n_prev = self.norm_action_prev[i]
        #     alpha = self.norm_action[i+4]
        #     alpha_prev = self.norm_action_prev[i+4]
        #     est_dot_n = (n-n_prev)/self.delta_t_action
        #     est_dot_alpha = (alpha-alpha_prev)/self.delta_t_action
            
        #     R_act -= self.w_abs_n * abs(n) - self.w_est_dot_n * abs(est_dot_n) - self.w_est_dot_alpha * abs(est_dot_alpha)
        
        reward = np.squeeze(self.R_gauss + self.R_AS_gauss)/1.4 + self.R_vel + self.R_thrust + self.R_thrust_d # + R_act
        
        return reward

    def _pub_actions(self, action):
        
        for i in range(len(self.act_ref)):
            self.action[i] = round(action[i] * self.max_thruster_rpm)
            self.action[i+4] = round(action[i+4] * self.max_azimuth_angle)
            self.act_ref[i].throttle_reference = round(action[i] * self.max_thruster_rpm)
            self.act_ref[i].angle_reference = round(action[i+4] * self.max_azimuth_angle)

        print(f"action: {action}")
        print(f"act_ref: {self.act_ref}")
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