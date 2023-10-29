# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:31:10 2022

@author: amire
"""

from stable_baselines3.common.env_checker import check_env
from IPP_drone_path_planner import droneEnv


env = droneEnv(observation_mode='cont',action_mode='cont', render=True)
# It will check your custom environment and output additional warnings if needed
check_env(env)