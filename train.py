import multiprocessing
import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from utils.callbacks import SaveOnBestTrainingRewardCallback

if __name__ == '__main__':
    env_id = 'CartPole-v1'
    cpu_count = multiprocessing.cpu_count()

    # We create a separate environment for evaluation
    eval_env = gym.make(env_id)

    # Create log dir
    log_dir = '/tmp/gym/'
    os.makedirs(log_dir, exist_ok=True)

    # wrap environment in Monitor
    eval_env = Monitor(eval_env, log_dir)

    # create vector environment for parallel processing
    vec_env = make_vec_env(env_id, n_envs=cpu_count)

    # Create the model
    model = PPO('MlpPolicy', vec_env, verbose=0)

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Mean reward (pre-training): {mean_reward} +/- {std_reward:.2f}')

    # Train the agent
    n_timesteps = 25000
    model.learn(n_timesteps, callback=SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir))
    model.save('models/cartpole')

    # Trained Agent, after training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Mean reward (post-training): {mean_reward} +/- {std_reward:.2f}')