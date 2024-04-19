# -*- coding: utf-8 -*-
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, A2C, DQN
import os
import time
from drone_environment import droneEnv
from stable_baselines3.common.env_checker import check_env
import sys

env=droneEnv(render=True, generate_world=True)
check_env(env)
# Load the trained agent
try:
	model_path = sys.argv[1]
except:
	print("enter model name (no .zip) as CML arg")
	exit()

model = A2C.load(model_path, env=env)

episodes=10

for ep in range(episodes):
	obs, _ = env.reset()
	# print(obs)

	running_reward=[]
	while not env.done:
		action, info= model.predict(obs)
		obs, reward, done, trunc, info = env.step(action)
		running_reward.append(reward)

	print(sum(running_reward))
