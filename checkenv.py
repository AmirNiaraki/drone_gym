# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:08:03 2022

@author: aniaraki
"""

from stable_baselines3.common.env_checker import check_env
from snakeenv import SnekEnv
from IPP_drone_path_planner import dronEnv



env = SnekEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)