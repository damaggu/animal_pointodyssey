import os
import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mediapy as media
import random

from gymnasium import Space
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE


MODEL_PATH = "envs/dog_model/dog.xml"
FRAME_SKIP = 25

def lin_decay(x: float | np.ndarray, min_zero: float, min_clip: float, max_clip: float, max_zero: float) -> float:
    assert min_zero < min_clip
    assert min_clip <= max_clip
    assert max_clip < max_zero
    lin_x = np.minimum((x - min_zero) / (min_clip - min_zero), 1 - (x - max_clip) / (max_zero - max_clip))
    return np.clip(lin_x, 0, 1)


class DogEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20
    }
    def __init__(
        self,
        # observation_space: Space,
        render_mode: str | None = None,
        width: int = ...,
        height: int = ...,
        camera_id: int | None = None,
        camera_name: str | None = None,
        default_camera_config: dict | None = None,
        from_pixels: bool = False,
        from_vectors: bool = True,
        **kwargs,
    ):
        self.count = 0
        model_path = os.path.abspath(MODEL_PATH)
        utils.EzPickle.__init__(self, model_path, **kwargs)
        MujocoEnv.__init__(self, model_path, FRAME_SKIP, observation_space=None, render_mode=render_mode, camera_id=0, **kwargs)
        obs_size = self.data.qpos.size - 1 + self.data.qvel.size + self.data.cfrc_ext[1:].size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    
    def step(self, action):
        x_before = self.data.qpos[0].copy()
        z_before = self.data.qpos[2].copy()
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0].copy()
        z_after = self.data.qpos[2].copy()

        x_vel = (x_after - x_before) / self.dt
        z_vel = (z_after - z_before) / self.dt
        obs = self._get_obs()
        reward = self._get_rew(x_vel, action)

        if self.render_mode == "human":
            self.render()

        info = {
            "x_pos": self.data.qpos[0],
            "y_pos": self.data.qpos[1]
        }
        # print(self.data.geom("leg1_l geom"))
        # if random.random() <= 0.001:
        # print(self.data.qpos[2], "bruh", )

        return obs, reward, False, False, info
    
    def _get_obs(self) -> np.ndarray:
        positions = self.data.qpos[1:].flatten()
        velocities = self.data.qvel
        contact = self.contact_forces().flatten()
        obs = np.concatenate([positions, velocities, contact])
        return obs
    
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext[1:]
        return np.arctan(raw_contact_forces / 100)
    
    def height_reward(self) -> float:
        # return lin_decay(self.data.qpos[2], 0.4, 0.9, 1.75, 3)
        return lin_decay(self.data.qpos[2], 0.4, 0.9, 1.3, 2.5)
    
    def ctrl_reward(self, action: np.ndarray) -> float:
        return np.linalg.norm(action)

    def _get_rew(self, vel: float, action: np.ndarray):
        vel_reward = vel
        health = self.height_reward() * 2
        ctrl_rew = -0.025 * self.ctrl_reward(action)
        return vel_reward + health + ctrl_rew

    def reset_model(self):
        lol = np.zeros_like(self.init_qpos)
        lol[2] += 0
        qpos = self.init_qpos + self.np_random.uniform(-0.25, 0.25, size=self.model.nq) + lol
        qvel = self.init_qvel + self.np_random.uniform(-0.25, 0.25, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def render(self):
        return super().render()


