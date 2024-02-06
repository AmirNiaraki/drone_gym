# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:38:41 2022

@author: amireniaraki
"""

from stable_baselines3 import PPO, A2C
import os
from drone_environment import droneEnv
import time
import sys


# get algorithm
#TODO

# saved models and training log paths
try:
	logdir = sys.argv[1]
except:
	logdir = f"Training/Logs/{int(time.time())}/"

try:
	modeldir = sys.argv[2]
except:
	modeldir = f"Training/Models/{int(time.time())}/"

print("log directory: " + logdir)
print("model directory: " + modeldir)

# make directories
# if not os.path.exists(modeldir):
# 	os.makedirs(modeldir)

# if not os.path.exists(logdir):
# 	os.makedirs(logdir)

# make environment
env = droneEnv('disc', render = False)
env.reset()

# make algorithm
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = logdir)
# model = A2C('MlpPolicy', env, verbose = 1, tensorboard_log = logdir)

print("\n\nPOLICY")
print(model.policy)
print("\n\nOBS SPACE")
print(env.observation_space)
print("\n\nACT SPACE")
print(env.action_space)
# exit()

timesteps = 1000
iterations = 1

for iters in range(0, iterations):
	print('iteration: ', iters)
	model.learn(total_timesteps = timesteps, reset_num_timesteps = False, tb_log_name = f"PPO")
	model.save(modeldir)