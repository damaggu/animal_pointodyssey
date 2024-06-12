from gymnasium.envs.registration import register

register(
     id="dog-v0",
     entry_point="envs.dog_env:DogEnv",
     max_episode_steps=1000,
)
register(
     id="mouse-v0",
     entry_point="envs.mouse_env:MouseEnv",
     max_episode_steps=1000,
)