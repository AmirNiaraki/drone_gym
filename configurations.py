# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:10:45 2022

@author: amire
"""

import numpy as np
import pandas as pd
from math import tan, radians, ceil

class Configs:

    def __init__(self):
        self.load_from_geotiff=True # if false then the world is generated with random seeds
        # self.geotiff_path='2021-7-13-padded.png'
        self.geotiff_path='images/sample.png'

        self.STATES_X=100
        self.STATES_Y=100
        self.STATES_Z=1
        self.init_state=[1,1,1]

# TODO: define aspect ratio from tan(FOV) and find the frame height based on AR and frame width
# basicaly frame H/W ~ Tan(FOV_Y)/Tan(FOV_X) 
        self.min_flight_height=80
        self.max_flight_height=80

        self.FOV_X=60/2 #degrees for halve of the field of view horizontaly
        self.FOV_Y=60/2 #degrees for halve of the field of view verticaly
        self.FRAME_W=100 #unit: pixels
        self.FRAME_H=100 #unit: pixels
        self.PADDING = self.calculate_padding(max(self.FOV_X,self.FOV_Y), self.max_flight_height)
        self.init_location=(self.PADDING,self.PADDING,self.min_flight_height)
        self.random_init_location=False
        
        if self.load_from_geotiff: ### this is where we load the world from the images dimensions
                import cv2
                img=cv2.imread(self.geotiff_path)
                self.wolrd_size_including_padding=[img.shape[1],img.shape[0]] 
                print('image height and width: ', img.shape[0], img.shape[1])
                ### WORLD_Xs and Ys will be updated inside the environment based on size of the image.
                self.WORLD_XS=[self.PADDING, self.wolrd_size_including_padding[0]-self.PADDING]
                self.WORLD_YS=[self.PADDING, self.wolrd_size_including_padding[1]-self.PADDING]
                self.WORLD_ZS=[self.min_flight_height,self.max_flight_height]
        else: # determine wold size HERE
                self.wolrd_size_including_padding=[2000,500]
                self.WORLD_XS=[self.PADDING, self.wolrd_size_including_padding[0]-self.PADDING]
                self.WORLD_YS=[self.PADDING, self.wolrd_size_including_padding[1]-self.PADDING]
                self.WORLD_ZS=[self.min_flight_height,self.max_flight_height]
        
        self.SEEDS=200
        self.square_size_range=(1,10)
        self.remove_redetected_from_world=False


        self.FULL_BATTERY=100.
        self.MAX_SPEED=5 #maximum allowed drone speed
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
        self.create_explored_map=True
        self.save_map_to_file=True
        self.show_location=False
        
    def calculate_padding(self, fov_degrees, drone_height):
                '''
                Returns number of pixels from the world that is visible from the centerpoint of 
                the frame.
                '''
                visible_pix=ceil(tan(radians(fov_degrees))*drone_height)
                return visible_pix
