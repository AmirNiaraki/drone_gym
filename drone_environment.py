# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:35:20 2022

@author: aniaraki


----> X
|
Y
"""
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import pandas as pd
from math import tan, radians, degrees, acos, sqrt, ceil
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


class droneEnv(gymnasium.Env):
    
    def __init__(self, observation_mode, action_mode,  render=False):
        # super(droneEnv, self).__init__()
        super().__init__()
        self.cfg=Configs()       
        self.observation_mode=observation_mode
        self.action_mode=action_mode
        self.render=render

        self.location=self.cfg.init_location.copy()

        if self.cfg.load_from_geotiff==True:
            self.world=self.load_geotiff()
        else:
            self.world=self.world_genertor() if self.cfg.is_world_generated==True else self.load_world()
        print(self.workd.shape)
### wind field = (wind_x, wind_y) m/s. with x pointing at east, and positive y pointing at south
        self.wind=(3.5, 0)       
        self.battery=self.cfg.FULL_BATTERY # [x,y,z,] m
        self.done=False
        self.reward=0
        self.total_reward=0
        self.step_count=0
        self.battery_inloop=True if self.cfg.battery_inloop else False

        self.drag_normalizer_coef=0.5
        
        self.action=[0,0,0]
        if self.observation_mode=='cont':
            self.observation_space = Box(low=0, high=255,
                                    shape=(self.cfg.FRAME_H, self.cfg.FRAME_W+1, 1), dtype=np.uint8) #NOTE: dtype has to change for non binary image
            # self.observation_space=Box(low=-2000, high=2000,
            #                            shape=(6,), dtype=np.float64)  
        if self.action_mode=='cont':
            self.action_space=Box(low=-self.cfg.MAX_SPEED, high=self.cfg.MAX_SPEED, shape=(3,), dtype=np.float64)
        if self.action_mode=='disc':
            self.action_space=Discrete(4)
        
        if self.observation_mode=='disc':
### action list for 2d: [0 ,1       ,2    ,3         ,4   ,5        ,6   ,7]
### action list for 2d: [up,up-right,right,right-down,down,down-left,left,left-top ]
            self.action_space=Box(low=-self.cfg.MAX_SPEED, high=self.cfg.MAX_SPEED, shape=(3,), dtype=np.float64)
            self.observation_space=Box(low=-2000, high=2000,
                                       shape=(6,), dtype=np.float64)
                   
### for getting the frame to the agent at all times
        time.sleep(0.01)
        self.thread=Thread(target=self.update_frame, args=(),daemon=True)
        self.thread.start()
        time.sleep(0.01)
        print('environment is initialized')        
        
    def update_frame (self):
        self.imager_thread_name=threading.current_thread()
        print('top of the thread')
        while self.done==False:
            self.visible_x=ceil(tan(radians(self.cfg.FOV_X))*2*self.location[2])
            self.visible_y=ceil(tan(radians(self.cfg.FOV_Y))*2*self.location[2])
            self.world_img=np.uint8((1-self.world)*255)
            ### take snap of the sim based on location [x,y,z]
            ### visible corners of FOV in the form boundaries= [y,y+frame_h,x,x+frame_w]
            self.boundaries=[int(-self.visible_y/2+self.location[1]),int(self.visible_y/2+self.location[1]), 
                             int(-self.visible_x/2+self.location[0]),int(self.visible_x/2+self.location[0])]

            # !!!! Sanity Check:!!!!
            # print('the boundaries are not proportional with altittude!!!:')

            # self.boundaries=[int(-self.cfg.FRAME_H/2+self.location[1]),int(self.cfg.FRAME_H/2+self.location[1]), int(-self.cfg.FRAME_W/2+self.location[0]),int(self.cfg.FRAME_W/2+self.location[0])]            
            self.boundaries=[int(-self.visible_y/2+self.location[1]),int(self.visible_y/2+self.location[1]), 
                             int(-self.visible_x/2+self.location[0]),int(self.visible_x/2+self.location[0])]
            crop=self.world_img[self.boundaries[0]:self.boundaries[1],self.boundaries[2]:self.boundaries[3]]

            resized=cv2.resize(crop, (self.cfg.FRAME_W, self.cfg.FRAME_H))    
            #         
            added_battery=self.concat_battery(resized)

            self.frame=added_battery
            
            if self.done==True:
                # print('done is true instide update_frame() trying to join')
                break
                
        # print('Frame Update stopping...  ',  self.imager_thread_name)

    def fetch_frame(self):
        return self.frame
    
    def fetch_anomaly(self):
        """
        This method is used to calculate the reward for the agent by counting the black pixels. 
        Additionally it sets the observed anomaly to 0 in the world.
        returns: 
        score (int); number of black pixels in the frame
        """
        observation=self.fetch_frame()
        # refers to image with no battery
        obs_with_no_battery=observation[0:self.cfg.FRAME_H,0:self.cfg.FRAME_W]
        # scor eis simply the number of black pixels
        score=self.cfg.FRAME_H*self.cfg.FRAME_W-np.sum(obs_with_no_battery/255, dtype=np.int32)
        ### removing the detected objects from the world!!!
        self.world [int(-self.visible_y/2+self.location[1]):int(self.visible_y/2+self.location[1]),
                    int(-self.visible_x/2+self.location[0]):int(self.visible_x/2+self.location[0])]=0
        return score




        
    def step(self, action, DISPLAY=False):
        '''
    defining navigation:     
    let's assume each step takes 1 second and moves the agent for =1 (s) * V (m/s)    
    the idea is that action is the absolute velocity. now if you have a heavy wind the cost will be higher
    but the absolute velocity won't change.
        '''
        self.action=action     
        self.reward=0

        if self.action_mode=='cont':
            # new location is assigned and reward of movement is calculated (move_cost())
            self.move_by_velocity()
        if self.action_mode=='disc':
            self.move_by_tile()
               
        if self.battery<1:
             self.reward-=10
             self.done=True
             self.close()
         
       
        if self.render==True and self.done==False:
            self.renderer()
            time.sleep(self.cfg.sleep_time) if self.cfg.sleep_time>0 else None  


        
        # exit()
        _score=self.fetch_anomaly()
        if _score>0:
            print('step',self.step_count, '\n this reward: ', _score, '\n')
        self.reward+=_score
        # self.reward+=self.fetch_anomaly()
        self.total_reward+=self.reward
        self.step_count+=1
        # print('STEP MTHD, count: ', self.step_count)

        # if self.fetch_anomaly()>0:
        #     print('step',self.step_count, '\n this reward: ', self.reward, '\n')
        #     print('Total rewards is:', self.total_reward)
        
        self.reward = float(self.reward)
        info={}
        
### defining observation
        if self.observation_mode=='cont':
            #making sure the image values are normalized
            observation=self.fetch_frame()
            observation=np.array(observation, dtype=np.uint8)
            observation=observation.reshape((self.cfg.FRAME_H, self.cfg.FRAME_W+1, 1))


            
        else:
            observation=[self.location[0], self.location[1], self.location[2], self.battery, self.wind[0], self.wind[1]]
            observation = np.array(observation , dtype=np.float64) 
        
        if DISPLAY==True:
            self.display_info()
        if self.cfg.MAX_STEPS<self.step_count:
            self.done=True
        # print('step count: ', self.step_count)
        # print(' x size on world: ', self.visible_x)
        # print( 'y size on world: ', self.visible_y)
        return observation, self.reward, self.done,  self.done, info # now needs both terminated and turncated booleans, passed done for both.



    def move_by_velocity(self):
        self.abs_velocity=self.action
        
        if  self.abs_velocity[0]<0:
            self.location[0]=max(self.location[0]+ self.abs_velocity[0], self.cfg.WORLD_XS[0])  
        else:
            self.location[0]=min(self.location[0]+ self.abs_velocity[0], self.cfg.WORLD_XS[1])
        
        if  self.abs_velocity[1]<0:
            self.location[1]=max(self.location[1]+ self.abs_velocity[1], self.cfg.WORLD_YS[0])  
        else:
            self.location[1]=min(self.location[1]+ self.abs_velocity[1], self.cfg.WORLD_YS[1])
        
        if  self.abs_velocity[2]<0:
            self.location[2]=max(self.location[2]+ self.abs_velocity[2],self.cfg. WORLD_ZS[0])  
        else:
            self.location[2]=min(self.location[2]+ self.abs_velocity[2], self.cfg.WORLD_ZS[1])
             
        self.reward =- self.move_cost()


    def renderer(self):
        try:

            ##world_crop
            _gray = cv2.cvtColor(self.world_img, cv2.COLOR_GRAY2BGR)
            img=cv2.rectangle(_gray, (self.boundaries[2],self.boundaries[0]),(self.boundaries[3],self.boundaries[1]),(255, 0, 0),3)
            img=cv2.putText(img, 'step ' + str(self.step_count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            img=cv2.putText(img, 'battery: '+ str(np.round(self.battery, 2)), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            img=cv2.putText(img, 'wind direction: '+ str(self.wind_angle), (50,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            img=cv2.putText(img, 'flight altitude: '+ str(self.location[2]), (50,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            #resize such that it fits the screen maintaining the aspect ratio
            AR=drone_crop_resized.shape[1]/drone_crop_resized.shape[0]
            img_resized=cv2.resize(img, (int(800*AR),800))
            cv2.imshow('World view', img)


            
        except:
            print('frame not available to render!')
            pass
        
        if cv2.waitKey(1)==ord('q'):
            print('Q hit:')
            self.done=True
            self.close()        
         
         
         
###############################################################################   
    def reset(self,**kwargs):
### for COARSE Coding there is an auxiliary function to get location from state
        # print('\n \n \n \n  reset happened!!! \n \n \n \n')
        print('number of active threads:', threading.active_count())        
        self.close()



        self.done = False
        time.sleep(0.01)
        self.thread=Thread(target=self.update_frame, args=(),daemon=True)
        self.thread.start()

        # self.location = self.cfg.init_location
        self.location = self.cfg.init_location.copy()

        self.world=self.world_genertor() if self.cfg.is_world_generated==True else self.load_world()
        self.battery=self.cfg.FULL_BATTERY # [x,y,z,] m
        self.reward=0
        self.total_reward=0
        self.step_count=0

        self.prev_reward=0
        self.score = 0 

### defining observation
        if self.observation_mode=='cont':
            observation=self.fetch_frame()
            observation=np.array(observation, dtype=np.uint8)
            observation=observation.reshape((self.cfg.FRAME_H, self.cfg.FRAME_W+1, 1))
        else:
            observation=[self.location[0], self.location[1], self.location[2], self.battery, self.wind[0], self.wind[1]]
            observation = np.array(observation , dtype=np.float64)
        info={}
        return observation, info
    def load_geotiff(self):
        geo=cv2.imread(self.cfg.geotiff_path, cv2.IMREAD_GRAYSCALE)
        print(f'geotiff loaded from file with size {geo.shape[0], geo.shape[1]}')
        return geo




    def world_genertor(self):
        ### the padded area of the world is were the drone cannot go to but may appear in the frame
        seeds=self.cfg.SEEDS
        size=(self.cfg.WORLD_YS[1]+self.cfg.PADDING,self.cfg.WORLD_XS[1]+self.cfg.PADDING)
        self.world=np.zeros(size, dtype=int)
        square_corners=[]
        for s in range(0,seeds):
             ### corner of each square corner=[x,y]
             corner=[random.randint(self.cfg.PADDING,self.cfg.WORLD_XS[1]),random.randint(self.cfg.PADDING,self.cfg.WORLD_YS[1])]
             ### list of all square corners
             square_corners.append(corner)
             square_size=random.randint(self.cfg.square_size_range[0],self.cfg.square_size_range[1])
             for i in range(0,square_size):
                 for j in range(0,square_size):
                     try:
                         self.world[corner[1]+j][corner[0]+i]=1
                     except:
                         pass
                     
        return self.world

    def load_world(self):
        _w=np.load(self.cfg.world_path)
        print('world loaded from file')
        return _w


    def loc_from_state(self):
        state_x_size=(self.cfg.WORLD_XS[1]-self.cfg.WORLD_XS[0])/self.cfg.STATES_X
        state_y_size=(self.cfg.WORLD_YS[1]-self.cfg.WORLD_YS[0])/self.cfg.STATES_Y
        state_z_size=(self.cfg.WORLD_YS[1]-self.cfg.WORLD_YS[0])/self.cfg.STATES_Z
        loc=[self.cfg.WORLD_XS[0]+state_x_size*(self.state[0]-1), self.cfg.WORLD_YS[0]+state_y_size*(self.state[1]-1), self.cfg.WORLD_ZS[0]+state_z_size*(self.state[2]-1)]
        ### returns a location=[x,y,z] where states=[1,1,1] corresponds to loc=[100,100,60] meteres
        return loc
    
    def close(self):
        # print('trying to close the env and destroy windows...')
        self.done=True
        time.sleep(0.1)
        self.imager_thread_name.join()
        cv2.destroyAllWindows()
        
        
### method receives frame as np array adds a column the end that represent battery level    
    def concat_battery(self, input_frame):
        full_pixels_count=int(self.cfg.FRAME_H*self.battery/100)
        full_pixels=np.zeros([full_pixels_count, 1])
        full_pixels.astype(int)
        empty_pixels=(np.zeros([self.cfg.FRAME_H-full_pixels_count,1])+1)*255
        empty_pixels.astype(int)

        battery_img=np.uint8(np.concatenate((empty_pixels, full_pixels)))
        self.output_frame=np.concatenate((input_frame, battery_img),axis=1)
        return self.output_frame
        
    def move_cost(self):
        #finding the relative angle of wind to drone
        # print('inside move_cost() method \n actions: ', self.action[0] , self.action[1], 'winds: ', self.wind[0], self.wind[1])
        self.wind_angle=degrees(acos((self.action[0]*self.wind[0]+self.action[1]*self.wind[1])
                                     /(sqrt(self.action[0]**2+self.action[1]**2)*sqrt(self.wind[0]**2+self.wind[1]**2))))
        #finding the relative velocity of wind to drone in absolute value
        self.relative_velocity=sqrt((self.action[0]-self.wind[0])**2+(self.action[1]-self.wind[1])**2)
        
        try: 
            self.drag=self.cfg.drag_table.iloc[round(self.wind_angle/45.), round((self.relative_velocity-12)/5)]*self.drag_normalizer_coef
        except:
            print('relative velocity/angles out of bounds.')
            # defualt drag
            self.drag=0.1*self.drag_normalizer_coef
        cost=self.drag*self.relative_velocity**2
        
### method to find the step cost based on drag force, for now everything costs 1
        # self.cost= self.c_d *((self.action[0]-self.wind[0])**2 + (self.action[1]-self.wind[1])**2)
        
        if self.battery_inloop==True:
            self.battery=max(0,self.battery-cost)
            return cost
        else:
            # print(self.battery)
            return 0
 
    def renderer(self):
        try:
            ##drone crop
            drone_crop=self.fetch_frame()
            drone_crop_resized=cv2.resize(drone_crop, (drone_crop.shape[1] // 3, drone_crop.shape[0] // 3))
            cv2.imshow('just fetched',drone_crop_resized)
            ##world crop
            _gray = cv2.cvtColor(self.world_img, cv2.COLOR_GRAY2BGR)
            img=cv2.rectangle(_gray, (self.boundaries[2],self.boundaries[0]),(self.boundaries[3],self.boundaries[1]),(255, 0, 0),3)
            img=cv2.putText(img, 'East WIND: '+ str(np.round(-self.wind[0],2)) +' North WIND:'+ str(np.round(self.wind[1],2)) , (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'step ' + str(self.step_count), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'battery: '+ str(np.round(self.battery, 2)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'Heading angle w.r.t wind: '+ str(np.round(self.wind_angle,2)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            img=cv2.putText(img, 'flight altitude: '+ str(np.round(self.location[2],2)), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            #resize such that it fits the screen maintaining the aspect ratio
            AR=drone_crop_resized.shape[1]/drone_crop_resized.shape[0]
            img_resized=cv2.resize(img, (int(800*AR),800))
            cv2.imshow('World view', img)
            
        except:
            print('frame not available for render!')
            pass
        
        if cv2.waitKey(1)==ord('q'):
            print('Q hit:')
            self.done=True
            self.close()        
         

    def display_info(self):
        print('==== INFO ==== \n')
        # print('step count: ', self.step_count)
        # print('battery level: ', self.battery)
        # print('current location: ', self.location)
        # print('current wind angle: ', self.wind_angle)
        # print('relative velocity: ', self.relative_velocity)
        print('step:', self.step_count, ' reward: ', self.reward)
        print('\n total reward: ', self.total_reward)

        
        print('========= \n')

        
        
        
   