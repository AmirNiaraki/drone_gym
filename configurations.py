# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:10:45 2022

@author: amire
"""

import numpy as np

class Configs:
    def __init__(self):
        self.STATES_X=100
        self.STATES_Y=100
        self.STATES_Z=1
        self.init_state=[1,1,1]
        self.init_location=[100,100,60]
        ### lets define a 1000 m * 250 m = 60 acres world
        ### lets assume the flight altitude can vary between 60 to 100 m
        ### The world generates square patches with sizes ranging between (1,10)
        self.WORLD_XS=[100,1900]
        self.WORLD_YS=[100,900]
        self.WORLD_ZS=[60,100]

        self.SEEDS=200
        self.square_size_range=(1,10)
        self.FRAME_W=350
        self.FRAME_H=250
        self.FOV_X=60/2 #degrees for halve of the field of view horizontaly
        self.FOV_Y=47/2 #degrees for halve of the field of view verticaly
        self.FULL_BATTERY=100
        self.MAX_SPEED=20 
        self.PADDING=100
        self.FPS=30
        self.OVERLAP=0.5
        ### the padded area of the world is were the drone cannot go to but may appear in the frame
        
cfg=Configs()