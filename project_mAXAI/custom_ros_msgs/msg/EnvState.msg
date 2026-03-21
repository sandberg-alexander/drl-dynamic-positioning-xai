# Environment state published by MilliAmpereEnv for the standalone viewer
float32[3] epsilon          # body-frame error [x_b, y_b, psi_deg]
float32[3] epsilon_ned      # NED-frame error [north, east, psi_deg]
float32[3] est_velocity     # estimated body velocity [u, v, r]
float32[3] target_pose      # [north, east, heading_deg]
float32[4] thrusters        # RPM per thruster
float32[4] angles           # azimuth degrees per thruster
float32[6] reward_components  # [gauss, AS_gauss, vel, thrust, thrust_d, angle_d]
float32 reward_total
uint32 time_step
uint32 episode
