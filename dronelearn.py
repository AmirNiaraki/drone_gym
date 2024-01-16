# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:38:41 2022

@author: amireniaraki
"""

from stable_baselines3 import PPO, A2C
import os
from drone_environment import droneEnv
import time



# TODO: Should be big function	
# Parameters: model/algo, timesteps, iterations
# Give the logs a more descriptive name, its kinda confusing as is


# saved models and training log paths
models_dir = f"Training/Models/{int(time.time())}/"
logdir = f"Training/Logs/{int(time.time())}/"

# make directories
if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# make environment
# env = droneEnv('cont', render = True)
env = droneEnv('disc', render = False)
env.reset()

# make algorithm
# model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = logdir)
model = A2C('MlpPolicy', env, verbose = 1, tensorboard_log = logdir)

timesteps = 1000000
iterations = 10

for iters in range(0, iterations):
	print('iteration: ', iters)
	model.learn(total_timesteps = timesteps, reset_num_timesteps = False, tb_log_name = f"A2C")
	model.save(f" {models_dir} / {timesteps * iters}")