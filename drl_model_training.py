from datetime import datetime
import json
import os
import gymnasium as gym
import divide21env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch

# create folders
LOGS_DIR = "logs"
BEST_MODEL_DIR = os.path.join(LOGS_DIR, "best_model")
RESULTS_DIR = os.path.join(LOGS_DIR, "results")
REWARDS_DIR = os.path.join(LOGS_DIR, "rewards")
dirs = [BEST_MODEL_DIR, RESULTS_DIR, REWARDS_DIR]
for dir in dirs:
    os.makedirs(dir, exist_ok=True)


# set the total interaction with the environment
TOTAL_TIMESTEPS = 1_000_000


# ensure environments have the same seed so the results are reproducible
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

# create environment
env = gym.make("Divide21-v0")
env.reset(seed=SEED)

# separate evaluation environment
eval_env = gym.make("Divide21-v0")
eval_env.reset(seed=SEED+1)

# setup evaluation callback (saves best model automatically)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=RESULTS_DIR,
    eval_freq=100,
    deterministic=True,
    render=False
)

def test_model(model, sessions=10):
    '''
    test a given trained model for a few sessions
    '''
    rewards_per_session = []
    for session in range(sessions):
        obs = env.reset(seed=SEED+session)
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # env.render()
        session_reward = {"session": session + 1, "reward": total_reward}
        rewards_per_session.append(session_reward)
    
    # create a rewards json file
    data = {"rewards per session": rewards_per_session}
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M%S")
    file = os.path.join(REWARDS_DIR, f'rewards_{date_time}.json')
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    # setup PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_divide21drl_tensorboard/"
    )

    # train the model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback
    )

    # save the final model
    model.save("divide21_drl_0")
    
    # test the model
    best_model_zip = os.path.join(BEST_MODEL_DIR, 'best_model.zip')
    best_model = PPO.load(best_model_zip)
    test_model(best_model)
    
    '''
    to visualize training progress, run the command bellow in a different terminal and open the link it shows:
        tensorboard --logdir ./ppo_divide21drl_tensorboard/
    '''

    # close environment
    env.close()
    eval_env.close()

