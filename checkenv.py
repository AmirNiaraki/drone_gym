# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:31:10 2022

@author: amire
"""

from stable_baselines3.common.env_checker import check_env
from drone_environment import droneEnv


env = droneEnv('cont')
# It will check your custom environment and output additional warnings if needed
check_env(env)