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
        self.STATES_X = 100
        self.STATES_Y = 100
        self.STATES_Z = 1
        self.init_state = [1, 1, 1]
        
        
        # Lets define a 1000 m * 250 m = 60 acres world
        # Lets assume the flight altitude can vary between 60 to 100 m
        

        # Number of seeds (for reward generation)
        self.SEEDS = 1000

        # Size of random rewards (TODO: Confirm this. Make it all caps?)
        # The world generates square patches with sizes ranging between (1,10)
        self.square_size_range = (1, 10)

        # Wind speed
        # Wind field = (wind_x, wind_y) m/s. with x pointing at east, and positive y pointing at south
        self.DEFAULT_WIND = (3.5, 0.0)

        # How many steps per episode
        self.MAX_STEPS = 10000
        



        ### DRONE CAMERA ###
        # Resolution of drone camera
        self.FRAME_W = 300
        self.FRAME_H = 200

        # Drone camera's FOV
        self.FOV_X = 60 / 2                 #degrees for halve of the field of view horizontaly
        self.FOV_Y = 40 / 2                 #degrees for halve of the field of view verticaly

        # Drone Camera FPS
        self.FPS = 30



        ### WORLD, REWARDS, and SIMULATION ###
        
        # Range of possible height values the drone can take
        self.WORLD_ZS = (100, 300)                 #World Boundaries (Z Axis)

        # The padded area of the world is were the drone cannot go to but may appear in the frame
        # Needed so that the drone's view isn't outide of the world
        # TODO: Needs to be calculated according the the drone's height and FOV. Move calculations from update_frame to here or droneEnv constructor
        self.PADDING_X = int(tan(radians(self.FOV_X)) * self.WORLD_ZS[1])
        self.PADDING_Y = int(tan(radians(self.FOV_Y)) * self.WORLD_ZS[1])

        # World dimensions
        self.WORLD_XS = (self.PADDING_X, 900)         #World Boundaries (X Axis)
        self.WORLD_YS = (self.PADDING_Y, 400)         #World Boundaries (Y Axis)



        ### DRONE ###

        # Drone's location starting location
        self.init_location = [self.PADDING_X,
                              self.PADDING_Y,
                              300.0]

        self.FULL_BATTERY = 100.0

        # Maximum allowed drone speed
        self.MAX_SPEED = 5

        # Percent overlap
        # Should be in range (0,1) exclusive
        self.OVERLAP = 0.90

        # Drone's drag
        self.drag_table = pd.read_csv('drag_dataset.csv', index_col=0)
        
cfg=Configs()
