# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:38:41 2022

@author: amireniaraki
"""

from stable_baselines3 import PPO, A2C
import os
from IPP_drone_path_planner import droneEnv
import time




models_dir = f"Training/Models/{int(time.time())}/"
logdir = f"Training/Logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = droneEnv('cont', render=False)
env.reset()


# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device='cuda')



TIMESTEPS = 100000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
    
    
