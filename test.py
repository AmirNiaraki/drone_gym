# -*- coding: utf-8 -*-
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
import os
import time

from stable_baselines3 import PPO, A2C
import os
import time
from IPP_drone_path_planner import droneEnv

env=droneEnv('disc', render=True)

# Load the trained agent
model_path = "Training/Models/1677789857/10000000"
model = PPO('MlpPolicy', env)
model.load(model_path)

episodes=10 

for ep in range(episodes):
    obs=env.reset()
    done=False
    ep_reward=[]
    while not done:
        action, _= model.predict(obs)
        obs, reward, done, info = env.step(action)
        ep_reward.append(reward)

    print(sum(ep_reward))
