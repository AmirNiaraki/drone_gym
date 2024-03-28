# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:10:45 2022

@author: amire
"""

import numpy as np
import pandas as pd
from math import tan, radians, degrees, acos, sqrt

class Configs:
    def __init__(self):
        # See loc_from_state in drone_environment
        # self.STATES_X = 100
        # self.STATES_Y = 100
        # self.STATES_Z = 1
        # self.init_state = [1, 1, 1]
        
        # Number of seeds (for reward generation)
        self.SEEDS = 500

        # The world generates square patches with sizes ranging between (1,10)
        self.square_size_range = (1, 10)

        # Wind speed
        # Wind field = (wind_x, wind_y) m/s. with x pointing at east, and positive y pointing at south
        self.DEFAULT_WIND = (0.001, 0.001)

        # How many steps per episode
        self.MAX_STEPS = 1000

        ### DRONE CAMERA ###
        # Resolution of drone camera
        # resolution such that minimum height does not upscale images
        self.FRAME_W = int(14.5 * 2)
        self.FRAME_H = int(9 * 2)

        # Drone camera's FOV
        self.FOV_X = 60 / 2                 #degrees for halve of the field of view horizontaly
        self.FOV_Y = 40 / 2                 #degrees for halve of the field of view verticaly

        # Drone Camera FPS
        # self.FPS = 30

        ### WORLD, REWARDS, and SIMULATION ###
        # Range of possible height values the drone can take
        # 80-400ft -> 25-125m
        self.WORLD_ZS = (25, 75)                 # World Boundaries (Z Axis)
        

        # The padded area of the world is were the drone cannot go to but may appear in the frame
        # Needed so that the drone's view isn't outide of the world
        # Calculated according the the drone's height and FOV. Move calculations from update_frame to here or droneEnv constructor
        self.PADDING_X = int(tan(radians(self.FOV_X)) * self.WORLD_ZS[1])
        self.PADDING_Y = int(tan(radians(self.FOV_Y)) * self.WORLD_ZS[1])

        # World dimensions
        self.WORLD_XS = (self.PADDING_X, 800)         #World Boundaries (X Axis)
        self.WORLD_YS = (self.PADDING_Y, 600)         #World Boundaries (Y Axis)

        ### DRONE ###
        self.FULL_BATTERY = 100.0

        # Maximum allowed drone speed
        self.MAX_SPEED = 20

        # Percent overlap (CC only)
        # Should be in range (0,1) exclusive
        self.OVERLAP = 0.90

        # Drone's drag
        self.drag_table = pd.read_csv('drag_dataset.csv', index_col=0)
        
cfg=Configs()
