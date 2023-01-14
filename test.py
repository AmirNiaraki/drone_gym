# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 09:43:09 2023

@author: amire
"""

from IPP_drone_path_planner import droneEnv

env = droneEnv('cont', render=False)

print('init location in class:', env.location, 'init location in cfg: ', env.cfg.init_location)
env.step([100,0,0])

print('init location in class:', env.location, 'init location in cfg: ', env.cfg.init_location)


