import gymnasium as gym

from gymnasium.wrappers import FlattenObservation, NormalizeObservation
import stable_baselines3
import sb3_contrib
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


from gymnasium.wrappers.monitoring import video_recorder
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

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
from typing import Callable
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
    "rodent-run-gaps": "dm_control/RodentRunGaps-v0",
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

class VecTrajRecorder(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        traj_folder: str,
        record_traj_trigger: Callable[[int], bool],
        traj_length: int = 200,
        name_prefix: str = "trajectory",
    ):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:
            metadata = temp_env.metadata

        self.env.metadata = metadata

        self.record_traj_trigger = record_traj_trigger
        self.traj_folder = os.path.abspath(traj_folder)
        # Create output folder if needed
        os.makedirs(self.traj_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.traj_length = traj_length

        self.recording = False
        self.recorded_frames = 0
        self.quats = []
        self.posl = []
        self.data = self.env.get_attr("data")[0]
        self.model = self.env.get_attr("model")[0]
        self.body_names = [self.model.body(i).name for i in range(self.model.nbody)]

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.start_traj_recorder()
        return obs

    def capture_state(self):
        quat, pos = self.data.xquat, self.data.xpos
        self.quats.append(quat.copy())
        self.posl.append(pos.copy())

    def start_traj_recorder(self) -> None:
        self.close_traj_recorder()

        traj_name = f"{self.name_prefix}-step-{self.step_id}-{self.traj_length}.npz"
        base_path = os.path.join(self.traj_folder, traj_name)
        self.path = base_path
        self.capture_state()
        self.recorded_frames = 1
        self.recording = True

    def _record_enabled(self) -> bool:
        return self.record_traj_trigger(self.step_id)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self.capture_state()
            self.recorded_frames += 1
            if self.recorded_frames > self.traj_length:
                print(f"Saving trajectory to {self.path}")
                self.close_traj_recorder()
        elif self._record_enabled():
            self.start_traj_recorder()

        return obs, rews, dones, infos


    def close_traj_recorder(self) -> None:
        self.recording = False
        self.recorded_frames = 1
        if len(self.quats) <= 1:
            return
        quats = self.quats[:-1]
        posl = self.posl[:-1]
        quats = np.stack(quats)
        posl = np.stack(posl)
        assert len(self.body_names) == quats.shape[1]
        final_traj = {name: np.concatenate((posl[:, i], quats[:, i]), axis=1) for i, name in enumerate(self.body_names)}
        np.savez_compressed(self.path, **final_traj)
        self.quats = []
        self.posl = []


    def close(self) -> None:
        VecEnvWrapper.close(self)
        self.close_traj_recorder()

    def __del__(self):
        self.close_traj_recorder()
def make_env(name: str, render_mode: str = None, **kwargs) -> gym.Env:
    if name == "rodent-bowl-escape-all" or name == "rodent-run-gaps":
        return RemoveZeroShapeObs(gym.make(REGISTERED_ENV_NAMES[args.env], render_mode=render_mode))

    if name in REGISTERED_ENV_NAMES:
        return gym.make(REGISTERED_ENV_NAMES[args.env], render_mode=render_mode)
    
    raise NotImplementedError(f"{name} not a implemented env.")


def parse_args(argparser: argparse.ArgumentParser) -> None:
    argparser.add_argument("--log-directory", type=str, default="logs")
    argparser.add_argument("--save-directory", type=str)
    argparser.add_argument("--model-directory", type=str, default="models")
    argparser.add_argument("--video-directory", type=str, default="videos")
    argparser.add_argument("--traj-directory", type=str, default="trajectories")
    argparser.add_argument("--record-traj", action='store_const', const=True, default=False)
    argparser.add_argument(
        "--env", type=str, default=list(REGISTERED_ENV_NAMES.keys())[0], choices=ENV_NAMES
    )
    argparser.add_argument("--checkpoint", type=str, default=None)
    argparser.add_argument("--num-timesteps", type=int, default=100_000)
    argparser.add_argument("--eval-freq", type=int, default=100_000)
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

    curr_dir = os.path.join(args.log_directory, args.save_directory)
    media.set_show_save_dir(os.path.join(curr_dir, args.video_directory))

    sb3_logger = configure(curr_dir, ["stdout", "json", "csv"])

    if args.checkpoint is not None:
        model = stable_baselines3.DDPG.load(args.checkpoint, env=env)
    else:
        policy_kwargs = {"net_arch": {"pi": [300, 200], "qf": [400, 300]}}
        model = stable_baselines3.DDPG(
            "MultiInputPolicy",
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
    print(args.record_traj)
    if args.record_traj:
        video_env = VecTrajRecorder(
            video_env,
            traj_folder=os.path.join(curr_dir, args.traj_directory),
            record_traj_trigger=lambda x: x % 10_000 < 1_000,
            traj_length=1_000,
        )

    eval_callback = EvalCallback(
        video_env,
        best_model_save_path=os.path.join(curr_dir, args.model_directory),
        log_path=curr_dir,
        eval_freq=args.eval_freq,
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
