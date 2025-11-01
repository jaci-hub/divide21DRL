'''
Name: Jacinto Jeje Matamba Quimua
Date: 10/30/2025

Train a Deep Reinforcement Learning (DRL) model using the RLlib Ray library in the Divide21Env
    System requirements for a smooth run: 6-8-core i7 + 16 GB RAM
'''

import os
import gymnasium as gym
import divide21env
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from gymnasium.spaces import Box
import numpy as np
from flattenDivide21EnvClass import FlattenDivide21Env

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"


# Define the log directory
LOG_DIR = "./rllib_divide21_tensorboard"
os.makedirs(LOG_DIR, exist_ok=True)
storage_dir = os.path.join(os.getcwd(), "rllib_divide21_tensorboard")


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
    "num_workers": 1, # ATTENTION: lower it to 0 if the system req above is not met
    "num_envs_per_worker": 1, # ATTENTION: comment this line if the system req above is not met
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
        metric="episode_reward_mean",
        mode="max",
        checkpoint_freq=50,
        keep_checkpoints_num=1
    )
    
    '''
    to visualise progress online, run the command below and open the link it shows:
        tensorboard --logdir ./rllib_divide21_tensorboard
    '''
    