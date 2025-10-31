import gym
import divide21env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# create environment
env = gym.make("Divide21-v0")

# separate evaluation environment
eval_env = gym.make("Divide21-v0")

# setup PPO model
model = PPO(
    "MlpPolicy",  # Multi-layer perceptron policy
    env,
    verbose=1,
    tensorboard_log="./ppo_divide21_tensorboard/"
)

# setup evaluation callback (saves best model automatically)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/results/",
    eval_freq=1000,
    deterministic=True,
    render=False
)

# train the model
model.learn(
    total_timesteps=100_000,
    callback=eval_callback
)

# save the final model
model.save("ppo_divide21_final")

# test the trained agent
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
