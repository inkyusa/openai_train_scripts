import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv,VecNormalize
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
import numpy as np
import os
import datetime
import tensorflow as tf

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
model_name = "ppo2_ratequad_{:%Y%m%d-%H-%M-%S}".format(datetime.datetime.now())
model_dir = model_dir+model_name
os.makedirs(model_dir, exist_ok=True)

env_id = 'QuadRate-v0'
num_cpu = 8 # Number of processes to use
n_timesteps = int(1e5)

env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False,clip_obs=10.)

# Custom MLP policy of two layers of size 32 each with tanh activation function
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128, 128])

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./log/",n_steps=2048, policy_kwargs=policy_kwargs, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10, ent_coef=0.0, learning_rate=3e-4, cliprange=0.2)
model.learn(n_timesteps, tb_log_name=model_name)
model.save(model_dir+"/model")
env.save_running_average(model_dir)
#mean_reward = evaluate(model, num_steps=10000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

