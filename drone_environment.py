# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:35:20 2022

@author: aniaraki

"""
import gymnasium as gym
from gymnasium.spaces import Box
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

class droneEnv(gym.Env):
    """
    Drone Environment class, extends gym.Env 
        ... many instance variables
    """
    def __init__(self, render=False, generate_world=True, wname=None, learn=True):
        """
        Constructor.

        """
        super().__init__()
        
        # cml args
        self.render = render
        self.generate_world = generate_world

        self.learn = learn

        # hardcoded
        if wname is None:
            self.world_name = "output_image.npy"# world file to be loaded if generate_world param is false
        else:
            self.world_name = wname
        self.cfg = Configs()                # config file for environment parameters
        self.wind = self.cfg.DEFAULT_WIND   # environment wind
        self.location = [0, 0, 0]           # location declataration. Initialization is in reset()

        # Coefficients
        self.move_coeff = 1.0               # scaling coefficient for movement penalty in step()
        self.detection_coeff = 3.0          # scaling coefficient for detection reward in step()
        self.explore_coeff = 0.01           # scaling coefficient for exploreation reward in step()

        # initialize everything else with reset()
        self.reset()
        
    # helper method (runs in separate thread)
    def update_frame(self):
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
            # in self.world, 1's are rewards. Get's converted to 0 to be displayed as black
            # in self.world, 0's are empty pixles. Get's converted to 255 to be displayed as white
            self.world_img = np.uint8((1 - self.world) * 255)

            # take snap of the sim based on location [x,y,z]
            # visible corners of FOV in the form boundaries= [y,y+frame_h,x,x+frame_w]
            print(f"self.location = {self.location}")
            self.boundaries = [int(-self.visible_y / 2 + self.location[1]),
                               int( self.visible_y / 2 + self.location[1]),
                               int(-self.visible_x / 2 + self.location[0]),
                               int( self.visible_x / 2 + self.location[0])]
            
            print(self.boundaries)

            # Crop the drone's view from the world
            crop = self.world_img[self.boundaries[0] : self.boundaries[1], self.boundaries[2] : self.boundaries[3]]
            
            # Resizes that crop to the resolution of the drone (downscale)
            # in self.frame,  255's are empty
            # in self.frame, 0's are rewards

            self.frame = cv2.resize(src=crop, dsize=(self.cfg.FRAME_W, self.cfg.FRAME_H), interpolation=cv2.INTER_NEAREST)

    # helper method
    def fetch_anomaly(self):
        """
        Checks self.frame for rewards (black pixels), and then removes them from the world (we think).

        Parameters: -

        Returns: score (# of black pixels)
        """
        # sum of black pixels (0's) after resizing visible area (see update_frame())
        score = self.cfg.FRAME_H * self.cfg.FRAME_W - np.count_nonzero(self.frame)
    
        # seen anamolies = # of 1's in self.world from visible area
        self.seen_anomalies += np.count_nonzero(self.world[self.boundaries[0] : self.boundaries[1], self.boundaries[2] : self.boundaries[3]])

        # Remove the detected objects from the world
        # Set everything equal to 0 (empty)
        self.world[self.boundaries[0] : self.boundaries[1],
                   self.boundaries[2] : self.boundaries[3]] = 0
        
        return score
    
    # helper method
    def explore(self):
        """
        Returns the number of unexplored pixels (0's) from new frame, then sets all pixels in frame to 1 (explored)
        """
        # number of unexplored pixles in visible area
        # (total visible area) - number of 1's in visible area
        score = abs(self.boundaries[0] - self.boundaries[1])*abs(self.boundaries[2] - self.boundaries[3]) - np.count_nonzero(self.explore_world[self.boundaries[0] : self.boundaries[1], self.boundaries[2] : self.boundaries[3]])

        # explore seen pixles
        self.explore_world[self.boundaries[0] : self.boundaries[1],
                           self.boundaries[2] : self.boundaries[3]] = 1

        return score

    # helper method        
    def get_obs(self):
        ###  EXPLORE FRAME | STATE VECTOR
        # resized version of explored world
        # area_covered = cv2.resize(self.explore_world[self.boundaries[0] : self.boundaries[1],
        #                                              self.boundaries[2] : self.boundaries[3]],
        #                           dsize=(self.cfg.FRAME_W, self.cfg.FRAME_H),
        #                           interpolation=cv2.INTER_NEAREST)
        # area_covered = np.array(area_covered)
        # # create normalized vector of state var
        # normalized_state_vector = np.array([[(self.location[0] - self.cfg.WORLD_XS[0])/(self.cfg.WORLD_XS[1] - self.cfg.WORLD_XS[0])],
        #                                     [(self.location[1] - self.cfg.WORLD_YS[0])/(self.cfg.WORLD_YS[1] - self.cfg.WORLD_YS[0])],
        #                                     [(self.location[2] - self.cfg.WORLD_ZS[0])/(self.cfg.WORLD_ZS[1] - self.cfg.WORLD_ZS[0])],
        #                                     [self.wind[0] / 10.0],
        #                                     [self.wind[1] / 10.0],
        #                                     [self.battery / 100.0]])
        # # concat zeros to get dimensions correct
        # normalized_state_vector = np.concatenate((normalized_state_vector, np.zeros((area_covered.shape[0] - normalized_state_vector.shape[0], 1))))
        # # create observation array=[area | state vec]
        # obs = np.concatenate((area_covered, normalized_state_vector), axis=1)

        ### EXPLORE WORLD | STATE VECTOR
        # # resize world
        # explore_resized = cv2.resize(self.explore_world, dsize=(10, 10), interpolation=cv2.INTER_NEAREST)

        # # norm vector
        # normalized_state_vector = np.array([[(self.location[0] - self.cfg.WORLD_XS[0])/(self.cfg.WORLD_XS[1] - self.cfg.WORLD_XS[0])],
        #                                     [(self.location[1] - self.cfg.WORLD_YS[0])/(self.cfg.WORLD_YS[1] - self.cfg.WORLD_YS[0])],
        #                                     [(self.location[2] - self.cfg.WORLD_ZS[0])/(self.cfg.WORLD_ZS[1] - self.cfg.WORLD_ZS[0])],
        #                                     [self.wind[0] / 10.0],
        #                                     [self.wind[1] / 10.0],
        #                                     [self.battery / 100.0]])
        # normalized_state_vector = np.concatenate((normalized_state_vector, np.zeros((explore_resized.shape[0] - normalized_state_vector.shape[0], 1))))

        # # concatenate entire explore array with the state vector
        # obs = np.concatenate((explore_resized, normalized_state_vector), axis=1)

        ## FRAME | STATE VECTOR
        normalized_state_vector = np.array([[(self.location[0] - self.cfg.WORLD_XS[0])/(self.cfg.WORLD_XS[1] - self.cfg.WORLD_XS[0])],
                                            [(self.location[1] - self.cfg.WORLD_YS[0])/(self.cfg.WORLD_YS[1] - self.cfg.WORLD_YS[0])],
                                            [(self.location[2] - self.cfg.WORLD_ZS[0])/(self.cfg.WORLD_ZS[1] - self.cfg.WORLD_ZS[0])],
                                            [self.wind[0] / 10.0],
                                            [self.wind[1] / 10.0],
                                            [self.battery / 100.0]])
        # concat zeros to get dimensions correct
        normalized_state_vector = np.concatenate((normalized_state_vector, np.zeros((self.frame.shape[0] - normalized_state_vector.shape[0], 1))))
        # create observation array=[area | state vec]
        obs = np.concatenate((self.frame // 255, normalized_state_vector), axis=1)

        ### NORMALIZED STATE VECTOR
        # obs = np.array([[(self.location[0] - self.cfg.WORLD_XS[0])/(self.cfg.WORLD_XS[1] - self.cfg.WORLD_XS[0])],
        #                 [(self.location[1] - self.cfg.WORLD_YS[0])/(self.cfg.WORLD_YS[1] - self.cfg.WORLD_YS[0])],
        #                 [(self.location[2] - self.cfg.WORLD_ZS[0])/(self.cfg.WORLD_ZS[1] - self.cfg.WORLD_ZS[0])],
        #                 [self.wind[0] / 10.0],
        #                 [self.wind[1] / 10.0],
        #                 [self.battery / 100.0]])

        return obs

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
        
        # calculate cost of movement
        cost = self.move_cost()

        # Subtract move_cost from reward and battery
        battery_coeff = 0.5
        self.battery = max(0, self.battery - cost * battery_coeff)

        # calculate and apply reward
        explore = self.explore_coeff *      self.explore()
        detection = self.detection_coeff *  self.fetch_anomaly()
        movement = self.move_coeff *        cost
        self.reward = detection + explore - movement
        ### DEBUGGING
        # print("detection: " + str(detection) + 
        #       "\tmovement: " + str(movement) + 
        #       "\texplore: " + str(explore) + 
        #       "\tseen-anom: " + str(self.seen_anomalies) +
        #       "\ttotal anom: " + str(self.total_world_anomalies))
        # time.sleep(1)
        ###
        self.total_reward += self.reward
               
        # increase step count
        self.step_count += 1

        # End simulation if the battery runs out
        if self.battery<1:
            ###
            # print("RAN OUT OF BATTERY")
            ###
            self.done = True
            self.close()

        # End simulation if 80% of the rewards are collected
        if self.seen_anomalies >= self.total_world_anomalies * 0.8:
            ###
            # print("COLLECTED 80% OF ANAMOLIES")
            ###
            self.done = True
            self.close()

        # End simulation if exceeding maximum allowed steps
        if self.cfg.MAX_STEPS < self.step_count:
            ###
            # print("EXCEEDED STEP COUNT")
            ###
            self.done=True
         
        # render new world
        if self.render and not self.done:
            self.renderer()
        time.sleep(0.001)
        
        # OBSERVATION
        observation = self.get_obs()

        if DISPLAY:
            self.display_info()
        
        info = {}
        truncated = False

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

        # random start location if running RL
        # if not learning, CC will set this before running each shape
        if self.learn:
            self.location[0] = np.random.uniform(self.cfg.WORLD_XS[0], self.cfg.WORLD_XS[1])
            self.location[1] = np.random.uniform(self.cfg.WORLD_YS[0], self.cfg.WORLD_YS[1])
            self.location[2] = np.random.uniform(self.cfg.WORLD_ZS[0], self.cfg.WORLD_ZS[1])
        else:
            #TODO: We need to set location somehow for CC_polygon

        # generate world
        if self.generate_world:
            # updates self.world
            self.world_genertor()
            
            # Save as numpy array and png
            # np.save('test_world', self.world)
            # cv2.imwrite("test_world.png", self.world * 255)
        else:
            # Load a saved world
            self.world = np.load(self.world_name)
        
        # generate array to describe explored pixels
        # 0 = unexplored
        # 1 = explored
        self.explore_world = np.zeros((self.world.shape[0], self.world.shape[1]))

        # sum total rewards (updates self.world_rewards)
        self.world_anomalies()

        # Define thread for getting the frame to the agent at all times
        time.sleep(0.01)
        self.thread = Thread(target=self.update_frame, args=(), daemon=True)
        self.thread.start()
        time.sleep(0.01)

        observation = self.get_obs()

        # observation space should be normalized between 0 and 1
        self.observation_space = Box(low=-1, high=1, shape=observation.shape, dtype=np.float64)

        # continuous action space: [x-velocity, y-velocity, z-velocity]
        self.action_space = Box(low=-self.cfg.MAX_SPEED, high=self.cfg.MAX_SPEED, shape=(3,), dtype=np.float64)

        info = {}
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

    def close(self):
        """
        End the simulation and clean stuff up

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
        
    def world_anomalies(self):
        # Effectivly the number of 1's in the world array
        self.total_world_anomalies = np.count_nonzero(self.world)
        self.seen_anomalies = 0

    def move_cost(self, hover_cost=0.1):
        """
        Calculates move cost. Depends on action taken, wind, drag (from drag_table).

        Parameters: -

        Returns: cost
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
            # print('relative velocity/angles out of bounds.')
            # defualt drag
            self.drag = self.drag_normalizer_coef

        # apply drag and hover
        cost = self.drag * self.relative_velocity ** 2 + hover_cost

        # Method to find the step cost based on drag force, for now everything costs 1
        # self.cost= self.c_d *((self.action[0]-self.wind[0])**2 + (self.action[1]-self.wind[1])**2)
        
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