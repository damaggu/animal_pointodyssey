import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
import stable_baselines3
import sb3_contrib
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from gymnasium.spaces.dict import Dict
from gymnasium import spaces
from gymnasium import ObservationWrapper, Env
from typing import Any

from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from dm_control.locomotion.examples import basic_rodent_2020
from dm_control.suite.wrappers import pixels 
import mediapy as media
import numpy as np

import os
import argparse
import datetime

import envs
from utils import log_to_dir

FPS = 40

REGISTERED_ENV_NAMES = {
    "acrobot-swingup": "dm_control/acrobot-swingup-v0",
    "dog-stand": "dm_control/dog-stand-v0",
    "dog-trot": "dm_control/dog-trot-v0",
    "dog-walk": "dm_control/dog-walk-v0",
    "humanoid-walk": "dm_control/humanoid-walk-v0",
    "humanoid-run": "dm_control/humanoid-run-v0",
    "humanoid": "Humanoid-v4",
    "custom-dog": "dog-v0",
    "custom-mouse": "mouse-v0",
    "rodent-escape-bowl": "dm_control/RodentEscapeBowl-v0",
}
ENV_NAMES = list(REGISTERED_ENV_NAMES.keys()) + ["rodent-bowl-escape-all",]


class DoubleToFloat(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.convert_keys = []
        for k in list(env.observation_space.keys()):
            if env.observation_space[k].dtype == np.float64:
                env.observation_space[k].dtype = np.dtype("float32")
                env.observation_space[k].low = env.observation_space[k].low.astype(np.float32)
                env.observation_space[k].high = env.observation_space[k].high.astype(np.float32)
                self.convert_keys.append(k)
        if env.action_space.dtype == np.float64:
            env.action_space.dtype = np.dtype("float32")
            env.action_space.low = env.action_space.low.astype(np.float32)
            env.action_space.high = env.action_space.high.astype(np.float32)

    def observation(self, observation: Any) -> Any:
        for k in self.convert_keys:
            observation[k] = observation[k].astype(np.float32)
        return observation

class RemoveCamera(ObservationWrapper):
    def __init__(self, env: Env, camera_key: str):
        super().__init__(env)
        obs_space = dict(env.observation_space)
        for k in list(obs_space.keys()):
            if k == camera_key:
                del obs_space[k]
        self.observation_space = Dict(obs_space)
        self.camera_key = camera_key
    
    def observation(self, observation: Any) -> Any:
        for k in list(observation.keys()):
            if k == self.camera_key:
                del observation[k]
        return observation

class RemoveZeroShapeObs(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        self.delete_keys = []
        self.no_shape_keys = []
        obs_space = dict(env.observation_space)
        for k in list(obs_space.keys()):
            if obs_space[k].shape == (0,):
                del obs_space[k]
                self.delete_keys.append(k)
            elif obs_space[k].shape == ():
                obs_space[k] = spaces.Box(-np.inf, np.inf, (1,), np.float64)
                self.no_shape_keys.append(k)

        self.observation_space = Dict(obs_space)
    
    def observation(self, observation: Any) -> Any:
        for k in list(observation.keys()):
            if k in self.delete_keys:
                del observation[k]
            elif k in self.no_shape_keys:
                observation[k] = observation[k].reshape(-1)
        return observation



def make_env(name: str, render_mode: str = None, **kwargs) -> gym.Env:
    if name in REGISTERED_ENV_NAMES:
        return gym.make(REGISTERED_ENV_NAMES[args.env], render_mode=render_mode)
    
    if name == "rodent-bowl-escape-all":
        return DmControlCompatibilityV0(basic_rodent_2020.rodent_escape_bowl(), render_mode=render_mode)
    
    raise NotImplementedError(f"{name} not a implemented env.")


def parse_args(argparser: argparse.ArgumentParser) -> None:
    argparser.add_argument("--log-directory", type=str, default="logs")
    argparser.add_argument("--save-directory", type=str)
    argparser.add_argument("--model-directory", type=str, default="models")
    argparser.add_argument("--video-directory", type=str, default="videos")
    argparser.add_argument(
        "--env", type=str, default=list(REGISTERED_ENV_NAMES.keys())[0], choices=ENV_NAMES
    )
    argparser.add_argument("--checkpoint", type=str, default=None)
    argparser.add_argument("--num-timesteps", type=int, default=100_000)
    argparser.add_argument("--lr", type=float, default=0.0003)


def get_video(model: BaseAlgorithm, video_name: str, vid_length: int) -> None:
    env = model.get_env()
    frames = []
    obs = env.reset()
    for _ in range(vid_length):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

    media.show_video(frames, title=video_name, fps=FPS)


def main(args: argparse.Namespace):
    env = make_env(args.env, render_mode="rgb_array")
    # env = FlattenObservation(env)
    env.metadata["render_fps"] = FPS

    # obs_space = dict(env.observation_space)
    # for k in list(obs_space.keys()):
    #     if obs_space[k].shape == (0,):
    #         del obs_space[k]
    # env.observation_space = Dict(obs_space)


    curr_dir = os.path.join(args.log_directory, args.save_directory)
    media.set_show_save_dir(os.path.join(curr_dir, args.video_directory))

    sb3_logger = configure(curr_dir, ["stdout", "json", "csv"])

    if args.checkpoint is not None:
        model = stable_baselines3.DDPG.load(args.checkpoint, env=env)
    else:
        policy_kwargs = {"net_arch": {"pi": [300, 200], "qf": [400, 300]}}
        model = stable_baselines3.DDPG(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            policy_kwargs=policy_kwargs,
            buffer_size=int(3.5e5),
            batch_size=64,
            tau=1e-3,
        )

    model.set_logger(sb3_logger)
    video_env = VecVideoRecorder(
        model.get_env(),
        video_folder=os.path.join(curr_dir, args.video_directory),
        record_video_trigger=lambda x: x % 10_000 < 1_000,
        video_length=1_000,
    )

    eval_callback = EvalCallback(
        video_env,
        best_model_save_path=os.path.join(curr_dir, args.model_directory),
        log_path=curr_dir,
        eval_freq=100_000,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path=os.path.join(curr_dir, args.model_directory),
        name_prefix="model_",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.num_timesteps,
        progress_bar=True,
        callback=[eval_callback, checkpoint_callback],
    )
    get_video(model, "final_video", vid_length=1_000)
    video_env.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    parse_args(argparser)
    args = argparser.parse_args()

    if args.save_directory is None:
        args.save_directory = f"{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}_{args.env}"
    main(args)
