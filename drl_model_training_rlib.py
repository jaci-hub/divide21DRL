import os
import gymnasium as gym
import divide21env
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from gymnasium.spaces import Box
import numpy as np

# Define the log directory
LOG_DIR = "./rllib_divide21_tensorboard"
os.makedirs(LOG_DIR, exist_ok=True)
storage_dir = os.path.join(os.getcwd(), "rllib_divide21_tensorboard")

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


# The env creator is still needed for RLlib to spawn environments in workers
def divide21_env_creator(env_config):
    import divide21env
    env = gym.make("Divide21-v0")
    return FlattenDivide21Env(env)

# Register with RLlib
register_env("Divide21-v0", divide21_env_creator)

# RLlib config
config = {
    "env": "Divide21-v0",
    "framework": "torch",
    "num_gpus": 0,
    "num_workers": 0,
    "log_level": "INFO",
    "seed": 1,
    "disable_env_checking": True,
}



if __name__ == "__main__":
    # Train using Tune with TensorBoard logging
    tune.run(
        PPO,
        stop={"timesteps_total": 1_000_000},
        config=config,
        storage_path=storage_dir,
        callbacks=[TBXLoggerCallback()],
        checkpoint_at_end=True,
    )
    
    '''
    to visualise progress online, run the command below and open the link it shows:
        tensorboard --logdir ./rllib_divide21_tensorboard
    '''
