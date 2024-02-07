# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:35:20 2022

@author: aniaraki

"""
from gym import Env
from gym.spaces import Box
import numpy as np
# import pandas as pd
from math import tan, radians, degrees, acos, sqrt
import random
# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.evaluation import evaluate_policy
import cv2
from threading import Thread
import threading
# from sys import exit
# import sys
import time
from configurations import Configs

class droneEnv(Env):
    """
    Drone Environment class, extends gym.Env 
        ... many instance variables
    """
    def __init__(self, render=False, generate_world=True):
        """
        Constructor.

        """
        super().__init__()
        
        # cml args
        self.render = render
        self.generate_world = generate_world

        # hardcoded
        self.world_name = "output_image.npy"
        self.cfg = Configs()
        self.wind = self.cfg.DEFAULT_WIND
        self.location = [0, 0, 0]

        # observation space: [location, wind, battery]
        self.observation_space = Box(low=-2000, high=2000, shape=(6,), dtype=np.float64)

        # action space: [x-velocity, y-velocity, z-velocity]
        self.action_space = Box(low=-self.cfg.MAX_SPEED, high=self.cfg.MAX_SPEED, shape=(3,), dtype=np.float64)

        # initialize everything else with reset()
        self.reset()
        
    def update_frame (self):
        """
        Updates self.frame based on variables from configurations determining what the drone can see

        Parameters: - 

        Returns: -
        """
        self.imager_thread_name = threading.current_thread()
        # print('top of the thread')
        while not self.done:
            # Recalculate drone's field of view
            self.visible_x = tan(radians(self.cfg.FOV_X)) * 2 * self.location[2]
            self.visible_y = tan(radians(self.cfg.FOV_Y)) * 2 * self.location[2]
            
            # black/white inversion for display (not calculations)
            self.world_img = np.uint8((1 - self.world) * 255)

            # take snap of the sim based on location [x,y,z]
            # visible corners of FOV in the form boundaries= [y,y+frame_h,x,x+frame_w]
            self.boundaries = [int(-self.visible_y / 2 + self.location[1]),
                               int( self.visible_y / 2 + self.location[1]),
                               int(-self.visible_x / 2 + self.location[0]),
                               int( self.visible_x / 2 + self.location[0])]

            # Crop the drone's view from the world
            crop = self.world_img[self.boundaries[0] : self.boundaries[1], self.boundaries[2] : self.boundaries[3]]
            
            # Resizes that crop to the resolution of the drone (upscale)
            resized = cv2.resize(crop, (self.cfg.FRAME_W, self.cfg.FRAME_H))
            # added_battery = self.concat_battery(resized)
            # self.frame = added_battery
            self.frame = resized

    def fetch_anomaly(self):
        """
        Checks self.frame for rewards (black pixels), and then removes them from the world (we think).

        Parameters: -

        Returns: score (# of black pixels)
        """
        # drone view picture with no battery
        nobat = (self.frame)[0 : self.cfg.FRAME_H, 0 : self.cfg.FRAME_W]
        
        # sum of black pixels
        # nobat / 255 normalizes values black pixels = 1
        score = self.cfg.FRAME_H * self.cfg.FRAME_W - np.sum(nobat / 255, dtype=np.int32)
        
        # Remove the detected objects from the world
        # Set everything equal to 0 because black and white is inverted in update_frame()
        self.world[int(-self.visible_y / 2 + self.location[1]) : int(self.visible_y / 2 + self.location[1]),
                   int(-self.visible_x / 2 + self.location[0]) : int(self.visible_x / 2 + self.location[0])] = 0
        
        return score
        
    def step(self, action, DISPLAY=False):
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
        
        # Move the Drone
        # x-direction
        if  self.action[0] < 0:
            self.location[0] = max(self.location[0] + self.action[0], self.cfg.WORLD_XS[0])  
        else:
            self.location[0] = min(self.location[0] + self.action[0], self.cfg.WORLD_XS[1])
        # y-direction
        if  self.action[1] < 0:
            self.location[1] = max(self.location[1] + self.action[1], self.cfg.WORLD_YS[0])  
        else:
            self.location[1] = min(self.location[1] + self.action[1], self.cfg.WORLD_YS[1])
        # z-direction
        if  self.action[2] < 0:
            self.location[2] = max(self.location[2] + self.action[2], self.cfg.WORLD_ZS[0])  
        else:
            self.location[2] = min(self.location[2] + self.action[2], self.cfg.WORLD_ZS[1])
             
        # Subtract move_cost from reward
        self.reward -= self.move_cost()
               
        # Check if battery is empty
        if self.battery<1:
             self.reward-=10
             self.done=True
             self.close()
         
        # render new world
        if self.render and not self.done:
            self.renderer()
        time.sleep(0.001)
        
        # Add any anomalies to reward with fetch_anomaly
        self.reward += self.fetch_anomaly()
        self.total_reward += self.reward
        self.reward.astype(np.float32)

        # increase step count
        self.step_count += 1
        
        # concatenate observation to be returned
        observation = [self.location[0], self.location[1], self.location[2], self.wind[0], self.wind[1], self.battery]
        observation = np.array(observation)

        if DISPLAY == True:
            self.display_info()

        # check if simulation is done
        if self.cfg.MAX_STEPS < self.step_count:
            self.done=True
        
        info = None
        truncated = None

        return observation, self.reward, self.done, truncated, info
            
    def reset(self, seed=None, options=None):
        """
        End and close current simulation. Start a new simulation with reinitialized vars.

        Amir: for COARSE Coding there is an auxiliary function to get location from state

        Parameters: -

        Returns: An 'observation' (np.array of location, wind, and battery)
        """
        # Close existing simulation
        self.close()

        # reset parameters
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.step_count = 0
        self.battery_inloop = True
        self.drag_normalizer_coef = 0.5
        self.action = [0, 0, 0]
        self.battery = 100.0

        # do not simplify
        self.location[0] = self.cfg.init_location[0]
        self.location[1] = self.cfg.init_location[1]
        self.location[2] = self.cfg.init_location[2]

        print(self.location)

        # generate world
        if self.generate_world:
            self.world_genertor() # sets self.world

            # Save as numpy array and png
            # np.save('test_world', self.world)
            # cv2.imwrite("test_world.png", self.world * 255)
        else:
            # Load a saved world
            self.world = np.load(self.world_name)

        # Define thread for getting the frame to the agent at all times
        time.sleep(0.01)
        self.thread=Thread(target=self.update_frame, args=(), daemon=True)
        self.thread.start()
        time.sleep(0.01)

        observation = [self.location[0], self.location[1], self.location[2], self.wind[0], self.wind[1], self.battery]
        observation = np.array(observation)

        info = None

        return observation, info
      
    def world_genertor(self):
        """
        Generates a new world for the drone based on size and # seeds specified in configurations.py

        Parameters: -
        """

        # The padded area of the world is were the drone cannot go to but may appear in the frame
        seeds = self.cfg.SEEDS

        # tuple representing dimensions of world
        size = (self.cfg.WORLD_YS[1] + self.cfg.PADDING_Y, self.cfg.WORLD_XS[1] + self.cfg.PADDING_X)

        # initialize world with zeros
        self.world = np.zeros(size, dtype=int)

        square_corners = []
        for s in range(seeds):
            # Corner of each square corner=[x,y] (PLACES REWARDS IN PADDING)
            # corner=[random.randint(-self.cfg.PADDING_X, self.cfg.WORLD_XS[1]) + self.cfg.PADDING_X,
            #         random.randint(-self.cfg.PADDING_Y, self.cfg.WORLD_YS[1]) + self.cfg.PADDING_Y]
             
            corner = [random.randint(0, self.cfg.WORLD_XS[1] - self.cfg.PADDING_X) + self.cfg.PADDING_X,
                      random.randint(0, self.cfg.WORLD_YS[1] - self.cfg.PADDING_Y) + self.cfg.PADDING_Y]

             # List of all square corners
            square_corners.append(corner)
            square_size = random.randint(self.cfg.square_size_range[0], self.cfg.square_size_range[1])
            for i in range(square_size):
                for j in range(square_size):
                    try:
                        self.world[corner[1] + j][corner[0] + i] = 1
                    except:
                        pass

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
        try:
            self.imager_thread_name.join()
        except:
            pass

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
        
        # Try and calculated the relative velocity of the drone wrt wind * normalizing factor
        try: 
            self.drag = self.cfg.drag_table.iloc[round(self.wind_angle / 45.), round((self.relative_velocity - 12) / 5)] * self.drag_normalizer_coef
        except:
            print('relative velocity/angles out of bounds.')
            # defualt drag
            self.drag = 0.1 * self.drag_normalizer_coef
        cost = self.drag * self.relative_velocity ** 2

        # Method to find the step cost based on drag force, for now everything costs 1
        # self.cost= self.c_d *((self.action[0]-self.wind[0])**2 + (self.action[1]-self.wind[1])**2)
        
        # apply cost to battery
        self.battery = max(0, self.battery - cost)
        
        # TESTING
        # print("action:\t" + str(self.action) + "\twind:\t" + str(self.wind) + "\tcost\t" + str(cost))

        return cost
 
    def renderer(self):
        """
        Renders the world, displayed into a window.

        Parameters: -

        Returns: -
        """
        try:
            cv2.imshow('just fetched', self.frame)

            # change to grayscale
            _gray = cv2.cvtColor(self.world_img, cv2.COLOR_GRAY2BGR)
            # ???
            img = cv2.rectangle(_gray, (self.boundaries[2], self.boundaries[0]), (self.boundaries[3], self.boundaries[1]), (255, 0, 0), 3)
            # add info text
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