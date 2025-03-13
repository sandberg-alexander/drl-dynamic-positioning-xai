from gymnasium.envs.registration import register

register(
    id="milliampereROS_env/MilliampereROS-v0",
    entry_point="milliampereROS_env.envs:MilliampereROSEnv",
)

register(
    id="milliampereROS_env/MilliampereRos4Thrusters-v0",
    entry_point="milliampereROS_env.envs:MilliampereRosEnv4Thrusters",
)