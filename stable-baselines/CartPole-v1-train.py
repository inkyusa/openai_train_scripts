import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
import numpy as np
import time, os, datetime

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
          obs = env.reset()
          episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    return mean_100ep_reward

model_dir = "./model/"
os.makedirs(model_dir, exist_ok=True)
model_name = "ppo2_cartpole_{:%Y%m%d-%H-%M-%S}".format(datetime.datetime.now())

env_id = 'CartPole-v1'
num_cpu = 8 # Number of processes to use
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./log/")
before_train_mean_reward = evaluate(model, num_steps=10000)

n_timesteps = 100000

# Multiprocessed RL Training
start_time = time.time()
model.learn(n_timesteps, tb_log_name=model_name)
total_time_multi = time.time() - start_time

print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))

# # Single Process RL Training
# single_process_model = PPO2(MlpPolicy, DummyVecEnv([lambda: gym.make(env_id)]), verbose=0)

# start_time = time.time()
# single_process_model.learn(n_timesteps)
# total_time_single = time.time() - start_time

# print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time_single, n_timesteps / total_time_single))

# print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))

model.save(model_dir+model_name)
mean_reward = evaluate(model, num_steps=10000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

