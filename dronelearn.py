# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:38:41 2022

@author: amireniaraki
"""

from stable_baselines3 import PPO, A2C, DQN
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import os
from drone_environment import droneEnv
import time
import sys

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


# make environment
env = droneEnv(render=False, generate_world=True)
check_env(env)

# make algorithm
model = A2C('MlpPolicy', env, verbose = 1, tensorboard_log = logdir)

# training hyperparameters
timesteps = 100000
iterations = 1

for iters in range(0, iterations):
	print('iteration: ', iters)
	model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=modeldir)
	model.save(modeldir)