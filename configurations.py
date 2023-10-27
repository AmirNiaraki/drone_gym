# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:10:45 2022

@author: amire
"""

import numpy as np
import pandas as pd

class Configs:
    def __init__(self):
        self.is_world_generated=False
        self.world_path='drawn_world.npy'

        self.STATES_X=100
        self.STATES_Y=100
        self.STATES_Z=1
        self.init_state=[1,1,1]

# TODO: define aspect ratio from tan(FOV) and find the frame height based on AR and frame width
        self.FOV_X=60/2 #degrees for halve of the field of view horizontaly
        self.FOV_Y=45/2 #degrees for halve of the field of view verticaly
        self.FRAME_W=70 #unit: pixels
        self.FRAME_H=50 #unit: pixels

        self.PADDING_flt = max(self.FRAME_H,self.FRAME_W)//2 #padding for the world: scalar value
        self.PADDING_flt = 150.0
        self.PADDING = int(self.PADDING_flt)
        self.init_location=[self.PADDING,self.PADDING,100.]
        ### lets define a 1000 m * 250 m = 60 acres world
        ### lets assume the flight altitude can vary between 60 to 100 m
        ### The world generates square patches with sizes ranging between (1,10)
        self.desired_world_size=[2000,500]
        self.WORLD_XS=[self.PADDING, self.desired_world_size[0]-self.PADDING]
        self.WORLD_YS=[self.PADDING, self.desired_world_size[1]-self.PADDING]
        self.WORLD_ZS=[60,100]

        self.SEEDS=20
        self.square_size_range=(1,10)


        self.FULL_BATTERY=100.
        self.MAX_SPEED=5 #maximum allowed drone speed
        self.PADDING=max(self.FRAME_H,self.FRAME_W)/2 #padding for the world
        self.FPS=30
        self.OVERLAP=0.5
        ### the default wind is blowing towards positive x (west to east) 
        self.DEFAULT_WIND=(3.5,0.)
        ### the padded area of the world is were the drone cannot go to but may appear in the frame
        self.drag_table=pd.read_csv('drag_dataset.csv', index_col=0)
        self.battery_inloop=False
        ### how many steps per episode
        self.MAX_STEPS=1000000
        self.sleep_time=0
        
# cfg=Configs()
