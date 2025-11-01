import gymnasium as gym
import divide21env
import numpy as np
from gymnasium.spaces import Box


class FlattenDivide21Env(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(self._get_flat_obs().shape[0],), dtype=np.float32
        )

    def _get_flat_obs(self):
        obs = self.env.reset()[0]  # if Gymnasium API returns (obs, info)
        flat = np.concatenate([
            obs['available_digits_per_index'].astype(np.float32),
            obs['dynamic_number'].astype(np.float32).flatten(),
            np.array([obs['player_turn']], dtype=np.float32),
            obs['players'].astype(np.float32).flatten(),
        ])
        return flat

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_flat_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_flat_obs(), reward, terminated, truncated, info