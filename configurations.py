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
        self.world_path='drawn_world_1.npy'
        self.load_from_geotiff=True
        # self.geotiff_path='2021-7-13-padded.png'
        self.geotiff_path='images/field1.tif'

        # self.geotiff_path='justB.png'

        self.STATES_X=100
        self.STATES_Y=100
        self.STATES_Z=1
        self.init_state=[1,1,1]

# TODO: define aspect ratio from tan(FOV) and find the frame height based on AR and frame width
        self.FOV_X=60/2 #degrees for halve of the field of view horizontaly
        self.FOV_Y=60/2 #degrees for halve of the field of view verticaly
        self.FRAME_W=1280 #unit: pixels
        self.FRAME_H=1280 #unit: pixels

        self.PADDING = 100
        self.init_location=[self.PADDING,self.PADDING,60.]
        self.random_init_location=False
        ### lets define a 1000 m * 250 m = 60 acres world
        ### lets assume the flight altitude can vary between 60 to 100 m
        ### The world generates square patches with sizes ranging between (1,10)
        self.wolrd_size_including_padding=[2000,500]
        self.WORLD_XS=[self.PADDING, self.wolrd_size_including_padding[0]-self.PADDING]
        self.WORLD_YS=[self.PADDING, self.wolrd_size_including_padding[1]-self.PADDING]
        self.WORLD_ZS=[60,100]

        self.SEEDS=200
        self.square_size_range=(1,10)
        self.remove_redetected_from_world=True


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
        self.MAX_STEPS=1000
        self.sleep_time=0
        
# cfg=Configs()
