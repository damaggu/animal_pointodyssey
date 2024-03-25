import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import mediapy as media
import numpy as np

import os
import argparse
import datetime

from utils import log_to_dir


ALL_ENVS = {
    "acrobot-swingup": gym.make("dm_control/acrobot-swingup-v0", render_mode="rgb_array")
}

def parse_args(argparser: argparse.ArgumentParser) -> None:
    argparser.add_argument("--log-directory", type=str, default="logs")
    argparser.add_argument("--save-directory", type=str)
    argparser.add_argument("--model-directory", type=str, default="models")
    argparser.add_argument("--video-directory", type=str, default="videos")
    argparser.add_argument("--env", type=str, default=list(ALL_ENVS.keys())[0], choices=list(ALL_ENVS.keys()))
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
    
    media.show_video(frames, title=video_name)
    

def main(args: argparse.Namespace):
    env = ALL_ENVS[args.env]
    env = NormalizeObservation(FlattenObservation(env))
    env.metadata["render_fps"] = 60

    curr_dir = os.path.join(args.log_directory, args.save_directory)
    media.set_show_save_dir(os.path.join(curr_dir, args.video_directory))

    sb3_logger = configure(curr_dir, ["stdout", "json", "csv"])

    model = stable_baselines3.PPO("MlpPolicy", env, learning_rate=args.lr)

    model.set_logger(sb3_logger)
    video_env = VecVideoRecorder(model.get_env(), video_folder=os.path.join(curr_dir, args.video_directory), record_video_trigger=lambda x: x % 10_000 < 1_000, video_length=1_000)

    eval_callback = EvalCallback(video_env, best_model_save_path=os.path.join(curr_dir, args.model_directory), log_path=curr_dir, eval_freq=10_000, deterministic=True,)
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=os.path.join(curr_dir, args.model_directory), name_prefix="model_", save_replay_buffer=True, save_vecnormalize=True)

    model.learn(total_timesteps=args.num_timesteps, progress_bar=True, callback=[eval_callback, checkpoint_callback])
    get_video(model, "final_video", vid_length=1_000)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    parse_args(argparser)
    args = argparser.parse_args()

    if args.save_directory is None:
        args.save_directory = f"{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}_{args.env}"
    main(args)
