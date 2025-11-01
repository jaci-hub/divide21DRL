'''
Name: Jacinto Jeje Matamba Quimua
Date: 10/30/2025

Test a Deep Reinforcement Learning (DRL) model using the RLlib Ray library in the Divide21Env
'''

import os
import gymnasium as gym
import divide21env
import numpy as np
from gymnasium.spaces import Box
from ray.rllib.algorithms.ppo import PPO
from flattenDivide21EnvClass import FlattenDivide21Env


if __name__ == "__main__":
    # Update this path to your actual best checkpoint
    CHECKPOINT_PATH = os.path.join( 
        "rllib_divide21_tensorboard", # add the rest of the path to the checkpoint!!!
    )

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    # Load trained PPO policy
    print(f"Loading PPO policy from checkpoint: {CHECKPOINT_PATH}")
    algo = PPO.from_checkpoint(CHECKPOINT_PATH)

    # Create environment (with same preprocessing as training)
    env = FlattenDivide21Env(gym.make("Divide21-v0"))

    # Run one or multiple evaluation episodes
    n_episodes = 3
    for ep in range(n_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        total_reward = 0
        step = 0

        while not (done or truncated):
            action = algo.compute_single_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Optional rendering (comment out for faster testing)
            # env.render()

        print(f"Episode {ep+1} finished after {step} steps with reward {total_reward}")

    env.close()
