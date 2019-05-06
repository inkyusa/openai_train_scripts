import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv,VecNormalize
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
import numpy as np
import time

#env_id = 'BallBouncingQuad-v0'
env_id = 'OveractuatedQuad-v0'
env = gym.make(env_id)
env = DummyVecEnv([lambda: env])
#env.load_running_average(model_path)
#model = PPO2.load(model_path+"model.pkl")
#mean_reward = evaluate(model, num_steps=10000)

obs = env.reset()
for i in range(10000):
    #action, _states = model.predict(obs)
    action = [0, 0, 0, 0,0,0,0,0]
    obs, rewards, dones, info = env.step(action)
    env.render()

