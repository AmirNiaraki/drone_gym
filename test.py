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
from drone_environment import droneEnv

env=droneEnv('disc', render=True, generate_world=False)

# Load the trained agent
model_path = "linetest_model"
model = PPO.load(model_path, env=env)

episodes=10

for ep in range(episodes):
	obs, _ = env.reset()
	# print("OBS")
	# print(obs)

	done=False
	running_reward=[]
	while not done:
		action, info= model.predict(obs)
		obs, reward, done, trunc, info = env.step(action)
		running_reward.append(reward)

	print(sum(running_reward))
