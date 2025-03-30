from gymnasium.envs.registration import register

register(
    id="milliAmpere1ROS_env/GridWorld-v0",
    entry_point="milliAmpere1ROS_env.envs:GridWorldEnv",
)

register(
    id="milliAmpere1ROS_env/MilliAmpere1ROS-v1",
    entry_point="milliAmpere1ROS_env.envs:MilliAmpere1RosEnvV1",
)

register(
    id="milliAmpere1ROS_env/MilliAmpere1ROS-v2",
    entry_point="milliAmpere1ROS_env.envs:MilliAmpere1RosEnvV2",
)
