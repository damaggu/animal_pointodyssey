from gymnasium.envs.registration import register


register(
     id="mouse-v0",
     entry_point="envs.mouse_env:MouseEnv",
     max_episode_steps=1000,
)

register(
     id="mouse-stand-v0",
     entry_point="envs.mouse_env:StandingMouseEnv",
     max_episode_steps=1000,
)

