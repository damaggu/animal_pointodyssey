import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
import stable_baselines3
import sb3_contrib
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import mediapy as media
import numpy as np

import os
import argparse
import datetime

from utils import log_to_dir

FPS = 30

ALL_ENV_NAMES = {
    "acrobot-swingup": "dm_control/acrobot-swingup-v0",
    "dog-stand": "dm_control/dog-stand-v0",
    "dog-trot": "dm_control/dog-trot-v0",
    "dog-walk": "dm_control/dog-walk-v0",
    "humanoid-walk": "dm_control/humanoid-walk-v0",
    "humanoid-run": "dm_control/humanoid-run-v0",
    "humanoid": "Humanoid-v4",
}


def parse_args(argparser: argparse.ArgumentParser) -> None:
    argparser.add_argument("--log-directory", type=str, default="logs")
    argparser.add_argument("--save-directory", type=str)
    argparser.add_argument("--model-directory", type=str, default="models")
    argparser.add_argument("--video-directory", type=str, default="videos")
    argparser.add_argument(
        "--env", type=str, default=list(ALL_ENV_NAMES.keys())[0], choices=list(ALL_ENV_NAMES.keys())
    )
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
    env = gym.make(ALL_ENV_NAMES[args.env], render_mode="rgb_array")
    env = FlattenObservation(env)
    env.metadata["render_fps"] = FPS

    curr_dir = os.path.join(args.log_directory, args.save_directory)
    media.set_show_save_dir(os.path.join(curr_dir, args.video_directory))

    sb3_logger = configure(curr_dir, ["stdout", "json", "csv"])
    policy_kwargs = {"net_arch": {"pi": [300, 200], "qf": [400, 300]}}

    model = stable_baselines3.DDPG(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        policy_kwargs=policy_kwargs,
        buffer_size=int(1e6),
        batch_size=64,
        tau=1e-3,
    )

    model.set_logger(sb3_logger)
    video_env = VecVideoRecorder(
        model.get_env(),
        video_folder=os.path.join(curr_dir, args.video_directory),
        record_video_trigger=lambda x: x % 1_000_000 < 1_000,
        video_length=1_000,
    )

    eval_callback = EvalCallback(
        video_env,
        best_model_save_path=os.path.join(curr_dir, args.model_directory),
        log_path=curr_dir,
        eval_freq=1_000_000,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=2_000_000,
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
