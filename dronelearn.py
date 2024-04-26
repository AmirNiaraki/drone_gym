# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:38:41 2022

@author: amireniaraki
"""

from stable_baselines3 import PPO, A2C, DQN
import os
from drone_environment import droneEnv
import time
from datetime import datetime

models_dir = f"Training/Models/{int(time.time())}/"
logdir = f"Training/Logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env = droneEnv('cont', render=True)
env = droneEnv(observation_mode='disc', action_mode='cont', render=True)
now = datetime.now()

# It will check your custom environment and output additional warnings if needed
# check_env(env)
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
print(date_time)
models_dir = f"Training/Models/{date_time}/"
logdir = f"Training/Logs/{date_time}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# env.reset()

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# model = DQN('MlpPolicy', env, verbose=1, buffer_size=5000, learning_starts=1000 ,tensorboard_log=logdir)

TIMESTEPS = 1000000
iters = 1

while iters<10:
    iters += 1
    print('iteration: ', iters)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
