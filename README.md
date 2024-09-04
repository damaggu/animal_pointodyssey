# How to Run

## Necessary Packages

Pretty self-explanatory, but you do need a package called `Shimmy` in order to import the dm-control environments into an OpenAI gym-type format. Stable Baselines 3 has some algorithms, SheepRL has some other algorithms, and `mediapy` is used for saving and displaying videos. Otherwise, the classic packages like `numpy`, `gymnasium`, etc. are all.

## Stable Baselines 3

`train.py` is the main training script, if using algorithms from Stable Baselines 3. 

Supported environments can be found in `train.py` under `REGISTERED_ENV_NAMES`. For example, to call OpenAI's Humanoid-v4 env, you would supply `train.py` with the argument `--env humanoid`.

Note on environments:
- specific dm-control environments may error with "... shape 0 ..." This is caused because some dm-control environment observations return NumPy arrays of shape `(0,)`, which causes Stable Baselines 3 to bug out. To resolve, I have included the environment wrapper `RemoveZeroShapeObs`, which is used by simply doing `env = RemoveZeroShapeObs(env)`.
- if using both vector and image inputs, lines instantiating the policy type must be changed from "MlpPolicy" to "MultiInputPolicy".
- Otherwise, if error `AttributeError: 'Box' object has no attribute 'spaces'` is returned then "MlpPolicy" must be used instead.
- custom environments can be supplied under the `envs` folder. Include your environment code similar to `envs/dog_env.py` or `envs/mouse_env.py` (which can be refactored), and then edit accordingly in `envs/__init__.py`.

To use different algorithms, there is no handy command-line argument for that yet. Instead, the code can be changed pretty straightforwardly on lines with `model = ...`

Other command-line arguments should be somewhat straightforward, as described under `parse_args`.

Here is an example run:

```sh
python train.py --env mouse-stand --log-directory ttemplogs --num-timesteps 10000000 --lr 0.001 --eval-freq 50000
````

## SheepRL

If using SheepRL algorithms, a lot of documentation is already online on their website/Github. Below I provide an example run and explain each argument:

```sh
sheeprl exp=dreamer_v2_benchmarks fabric.accelerator=cuda fabric.devices=2 fabric.precision=16-mixed 'algo.cnn_keys.encoder=[rgb]' 'algo.cnn_keys.decoder=[rgb]' 'algo.mlp_keys.encoder=[state]' env.wrapper.from_vectors=True env.wrapper.from_pixels=True env=dmc env.id=humanoid_walk env.num_envs=10 env.max_episode_steps=-1 algo.world_model.kl_regularizer=2 algo.actor.objective_mix=0 algo.actor.ent_coef=1e-5 env.wrapper.domain_name=humanoid env.wrapper.task_name=walk algo.total_steps=10000000
```

Calls with template parameters from `dreamer_v2_benchmarks`, sets the encoder inputs (image vs vector inputs), the environment to use, and some other algorithm parameters.
