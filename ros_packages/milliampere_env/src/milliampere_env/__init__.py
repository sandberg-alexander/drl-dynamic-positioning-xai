"""milliampere_env -- consolidated gymnasium environment for milliAmpere1 DP."""

from gymnasium.envs.registration import register

register(
    id="MilliAmpere1-v1",
    entry_point="milliampere_env.milliampere_env:MilliAmpereEnv",
)
