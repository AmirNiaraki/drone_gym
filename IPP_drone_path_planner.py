# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:35:20 2022

@author: aniaraki

"""
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import pandas as pd
from math import tan, radians, degrees, acos, sqrt
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
from threading import Thread
import threading
from sys import exit
import sys
import time
from configurations import Configs
import pickle

class droneEnv(gym.Env):
    """
    Drone Environment class, extends gym.Env 
        ... many instance variables
    """
    def __init__(self, name, render=False):
        """
        Constructor.

        Parameters: name ("cont" or "disc"), renderer (default False)
        """
        # super(droneEnv, self).__init__()
        super().__init__()
        self.cfg = Configs()       
        self.name = name
        self.render = render

        # self.location=self.cfg.init_location
        self.location=[100., 100., 60.]

        # Generates a new world
        # self.world=self.world_genertor()
        # np.save('test_world_1', self.world)

        # Load a saved world
        self.world=np.load('test_world_1.npy')

        # Wind field = (wind_x, wind_y) m/s. with x pointing at east, and positive y pointing at south
        self.wind = (3.5, 0)       
        self.battery = self.cfg.FULL_BATTERY # [x,y,z,] m
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.step_count = 0
        self.battery_inloop = True
        self.drag_normalizer_coef = 0.5
        self.action = [0, 0, 0]

        # define observation_space (cont/disc)
        if self.name == 'cont':
            # self.observation_space = Box(low=0, high=255,
            # shape=(self.cfg.FRAME_H, self.cfg.FRAME_W+1), dtype=np.uint8)
            self.observation_space = Box(low=-2000, high=2000, shape=(6,), dtype=np.float64)
            self.action_space = Box(low=-self.cfg.MAX_SPEED, high=self.cfg.MAX_SPEED, shape=(3,), dtype=np.float64)

        elif self.name == 'disc':
            # action list for 2d: [0 ,1       ,2    ,3         ,4   ,5        ,6   ,7]
            # action list for 2d: [up,up-right,right,right-down,down,down-left,left,left-top ]
            self.action_space = Box(low=-self.cfg.MAX_SPEED, high=self.cfg.MAX_SPEED, shape=(3,), dtype=np.float64)
            self.observation_space = Box(low=-2000, high=2000, shape=(6,), dtype=np.float64)
                   
        # Define thread for getting the frame to the agent at all times
        time.sleep(0.01)
        self.thread=Thread(target=self.update_frame, args=(),daemon=True)
        self.thread.start()
        time.sleep(0.01)
        print('environment is initialized')        
        
    
    def update_frame (self):
        """
        Does some multithreading stuff...
        Is called once during constructor, and once during reset().
        Calculates drone's frame size, based on FOV and height.
        Concatenates battery to edge of the frame.
        Defines self.frame

        TODO: Should probably be part of the constructor

        Parameters: - 

        Returns: -
        """
        self.imager_thread_name = threading.current_thread()
        print('top of the thread')
        while self.done == False:
            self.visible_x = tan(radians(self.cfg.FOV_X)) * 2 * self.location[2]
            self.visible_y = tan(radians(self.cfg.FOV_Y)) * 2 * self.location[2]
            
            # color inversion
            self.world_img = np.uint8((1 - self.world) * 255)

            # take snap of the sim based on location [x,y,z]
            # visible corners of FOV in the form boundaries= [y,y+frame_h,x,x+frame_w]
            self.boundaries = [int(-self.visible_y/2+self.location[1]),
                               int(self.visible_y/2+self.location[1]),
                               int(-self.visible_x/2+self.location[0]),
                               int(self.visible_x/2+self.location[0])]

            # battery pixels stuff
            crop = self.world_img[self.boundaries[0] : self.boundaries[1], self.boundaries[2] : self.boundaries[3]]
            resized = cv2.resize(crop, (self.cfg.FRAME_W, self.cfg.FRAME_H))
            added_battery = self.concat_battery(resized)
            self.frame = added_battery
            
            if self.done == True:
                break

    def fetch_frame(self):
        """
        Parameters: -

        Returns: frame (what the drone is currently seeing)
        """
        return self.frame
    
    def fetch_anomaly(self):
        """
        Checks self.frame for rewards (black pixels), and then removes them from the world (we think).

        Parameters: -

        Returns: score (# of black pixels)
        """
        observation = self.fetch_frame()

        # drone view picture with no battery
        nobat = observation[0 : self.cfg.FRAME_H, 0 : self.cfg.FRAME_W]
        
        # number of pixels - sum of black pixels
        score = self.cfg.FRAME_H * self.cfg.FRAME_W - np.sum(nobat / 255, dtype=np.int32)
        
        # Remove the detected objects from the world
        # Set everything equal to 0 because black and white is inverted in update_frame()
        self.world[int(-self.visible_y / 2 + self.location[1]) : int(self.visible_y / 2 + self.location[1]),
                   int(-self.visible_x / 2 + self.location[0]) : int(self.visible_x / 2 + self.location[0])] = 0
        
        return score
        
    def step(self, action, DISPLAY=True):
        '''
        Moves drone and calculates reward based on seen anomalies and cost of move. Can print info to terminal if DISPLAY=True.
        
        Amir's Notes:
        defining navigation:     
        let's assume each step takes 1 second and moves the agent for =1 (s) * V (m/s)    
        the idea is that action is the absolute velocity. now if you have a heavy wind the cost will be higher
        but the absolute velocity won't change.

        Parameters: action (tuple of change in direction taken by drone each step), DISPLAY (default True, prints info to the terminal)

        Returns: observation (array of location, battery, wind), self.reward, self.info
        '''
        self.action = action
        self.reward = 0
        self.abs_velocity = self.action
        
        # Move the Drone
        # x-direction
        if  self.abs_velocity[0] < 0:
            self.location[0] = max(self.location[0] + self.abs_velocity[0], self.cfg.WORLD_XS[0])  
        else:
            self.location[0] = min(self.location[0] + self.abs_velocity[0], self.cfg.WORLD_XS[1])
        
        # y-direction
        if  self.abs_velocity[1] < 0:
            self.location[1] = max(self.location[1] + self.abs_velocity[1], self.cfg.WORLD_YS[0])  
        else:
            self.location[1] = min(self.location[1] + self.abs_velocity[1], self.cfg.WORLD_YS[1])
        
        # z-direction
        if  self.abs_velocity[2] < 0:
            self.location[2] = max(self.location[2] + self.abs_velocity[2],self.cfg. WORLD_ZS[0])  
        else:
            self.location[2] = min(self.location[2] + self.abs_velocity[2], self.cfg.WORLD_ZS[1])
             
        # Subtract move_cost from reward
        self.reward -= self.move_cost()
               
        # Check if battery is empty
        if self.battery<1:
             self.reward-=10
             self.done=True
             #self.close()
         
        # Renderer?
        if self.render == True and self.done == False:
            self.renderer()
        time.sleep(0.001)  # Probably have to wait for renderer or something
        
        # Add any anomalies to reward with fetch_anomaly.
        self.reward += self.fetch_anomaly()
        self.total_reward += self.reward

        self.step_count += 1
        # if self.fetch_anomaly()>0:
        #     print('step',self.step_count, '\n this reward: ', self.reward, '\n')
        #     print('Total rewards is:', self.total_reward)
        
        # ???
        self.reward.astype(np.float32)
        info = {}
        
        # define observation        
        # observation=self.fetch_frame()        
        observation = [self.location[0], self.location[1], self.location[2], self.battery, self.wind[0], self.wind[1]]
        observation = np.array(observation) 

        if DISPLAY == True:
            self.display_info()
        if self.cfg.MAX_STEPS < self.step_count:
            self.done=True
        
        return observation, self.reward, self.done, info
         
         
###############################################################################   
    def reset(self):
        """
        End and close current simulation. Start a new simulation with reinitialized vars.

        Amir: for COARSE Coding there is an auxiliary function to get location from state

        Parameters: -

        Returns: An 'observation' (np.array of location, wind, and battery)
        """
        print('number of active threads:', threading.active_count())        
        self.close()

        # TODO: Can't we just call constructor on self, and replace all of this?
        # Start new simulation
        self.done = False

        # Start new thread in update_frame
        time.sleep(0.01)
        self.thread = Thread(target=self.update_frame, args=(),daemon=True)
        self.thread.start()

        self.location = self.cfg.init_location
        # self.location = [100.,100.,60.]

        # TODO: remove this (need to figure out why it's here in the first place)
        # self.world=self.world_genertor()
        self.world = np.load('test_world_1.npy')

        self.battery = self.cfg.FULL_BATTERY # [x,y,z,] m
        self.reward = 0
        self.total_reward = 0
        self.step_count = 0

        self.prev_reward = 0
        self.score = 0 

        observation = [self.location[0],self.location[1],self.location[2], self.battery, self.wind[0], self.wind[1]]
        observation = np.array(observation) 
        
        return observation
      
    def world_genertor(self):
        """
        Generates a new world for the drone based on size and # seeds specified in configurations.py

        Parameters: -

        Returns: self.world (TODO: why? just update self.world?)
        """

        # The padded area of the world is were the drone cannot go to but may appear in the frame
        seeds = self.cfg.SEEDS
        size = (self.cfg.WORLD_YS[1] + self.cfg.PADDING, self.cfg.WORLD_XS[1] + self.cfg.PADDING)
        self.world = np.zeros(size, dtype=int)
        square_corners = []
        for s in range(0, seeds):
             # Corner of each square corner=[x,y]
             corner=[random.randint(self.cfg.PADDING, self.cfg.WORLD_XS[1]), random.randint(self.cfg.PADDING, self.cfg.WORLD_YS[1])]
             # List of all square corners
             square_corners.append(corner)
             square_size = random.randint(self.cfg.square_size_range[0], self.cfg.square_size_range[1])
             for i in range(0,square_size):
                 for j in range(0,square_size):
                     try:
                         self.world[corner[1] + j][corner[0] + i] = 1
                     except:
                         pass
                     
        return self.world

    def loc_from_state(self):
        """
        I don't understand the purpose of this method, why are there two different ways to get location?

        Parameters: -

        Returns: Amir: location=[x,y,z] where states=[1,1,1] corresponds to loc=[100,100,60] meteres
        """
        state_x_size = (self.cfg.WORLD_XS[1] - self.cfg.WORLD_XS[0]) / self.cfg.STATES_X
        state_y_size = (self.cfg.WORLD_YS[1] - self.cfg.WORLD_YS[0]) / self.cfg.STATES_Y
        state_z_size = (self.cfg.WORLD_YS[1] - self.cfg.WORLD_YS[0]) / self.cfg.STATES_Z

        loc = [self.cfg.WORLD_XS[0] + state_x_size * (self.state[0] - 1),
               self.cfg.WORLD_YS[0] + state_y_size * (self.state[1] - 1),
               self.cfg.WORLD_ZS[0] + state_z_size * (self.state[2] - 1)]
        
        return loc
    
    def close(self):
        """
        End the simulation and clean stuff up

        TODO: I think this method needs to override close from gym.env

        Parameters: -

        Returns: -
        """
        self.done = True

        # Closes active thread
        time.sleep(0.1) 
        self.imager_thread_name.join()

        # Close windows
        cv2.destroyAllWindows()
        
    def concat_battery(self, input_frame):
        """
        Amir: Receives frame as np array, adds a column the end that represent battery level

        Parameters: input_frame

        Returns: self.output_frame (input_frame with battery info, TODO: why self?)
        """
        full_pixels_count = int(self.cfg.FRAME_H * self.battery / 100)
        full_pixels = np.zeros([full_pixels_count, 1])
        full_pixels.astype(int)
        empty_pixels = (np.zeros([self.cfg.FRAME_H - full_pixels_count,1]) + 1) * 255
        empty_pixels.astype(int)

        battery_img = np.uint8(np.concatenate((empty_pixels, full_pixels)))
        self.output_frame = np.concatenate((input_frame, battery_img),axis = 1)
        return self.output_frame
        
    def move_cost(self):
        """
        Calculates move cost. Depends on action taken, wind, drag (from drag_table). Applies cost to the battery.

        Parameters: -

        Returns: self.cost (TODO: again, why self?)
        """
        # print('inside move_cost() method \n actions: ', self.action[0] , self.action[1], 'winds: ', self.wind[0], self.wind[1])
        # Finding the relative angle of wind to drone
        self.wind_angle = degrees(acos((self.action[0] * self.wind[0] + self.action[1] * self.wind[1]) /
                                       (sqrt(self.action[0] ** 2 + self.action[1] ** 2) * sqrt(self.wind[0] ** 2 + self.wind[1] ** 2))))
        
        # Finding the relative velocity of wind to drone in absolute value
        self.relative_velocity = sqrt((self.action[0] - self.wind[0]) ** 2 + (self.action[1] - self.wind[1]) ** 2)
        
        try: 
            self.drag = self.cfg.drag_table.iloc[round(self.wind_angle / 45.), round((self.relative_velocity - 12) / 5)] * self.drag_normalizer_coef
        except:
            print('relative velocity/angles out of bounds.')
            # defualt drag
            self.drag = 0.1 * self.drag_normalizer_coef
        self.cost = self.drag * self.relative_velocity ** 2

        # Method to find the step cost based on drag force, for now everything costs 1
        # self.cost= self.c_d *((self.action[0]-self.wind[0])**2 + (self.action[1]-self.wind[1])**2)
        
        ########### UNCOMMENT TO APPLY COST TO BATTERY ###########
        # if self.battery_inloop==True:
        #     self.battery = max(0, self.battery - self.cost)
        #     print(self.battery)
        #
        # return self.cost

        # Comment out to apply cost
        return 0
 
    def renderer(self):
        """
        Renders the world, displayed into a window.

        Parameters: -

        Returns: -
        """
        try:
            cv2.imshow('just fetched', self.fetch_frame())
            _gray = cv2.cvtColor(self.world_img, cv2.COLOR_GRAY2BGR)
            img = cv2.rectangle(_gray, (self.boundaries[2], self.boundaries[0]), (self.boundaries[3], self.boundaries[1]), (255, 0, 0), 3)
            img = cv2.putText(img, 'East WIND: '+ str(np.round(-self.wind[0],2)) +' North WIND:'+ str(np.round(self.wind[1],2)) , (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            img = cv2.putText(img, 'step ' + str(self.step_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            img = cv2.putText(img, 'battery: '+ str(np.round(self.battery, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            img = cv2.putText(img, 'Heading angle w.r.t wind: '+ str(np.round(self.wind_angle, 2)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            img = cv2.putText(img, 'flight altitude: '+ str(np.round(self.location[2],2)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('World view', img)

            
        except:
            print('frame not available for render!')
            pass
        
        if cv2.waitKey(1)==ord('q'):
            print('Q hit:')
            self.done=True
            self.close()
         
    
    def display_info(self):
        """
        Prints info to terminal, if DISPLAY=True in step method.

        Parameters: -

        Returns: -
        """
        print('==== INFO ==== \n')
        # print('step count: ', self.step_count)
        # print('battery level: ', self.battery)
        # print('current location: ', self.location)
        # print('current wind angle: ', self.wind_angle)
        # print('relative velocity: ', self.relative_velocity)
        print('step:', self.step_count, ' reward: ', self.reward)
        print('\n total reward: ', self.total_reward)

        
        print('========= \n')