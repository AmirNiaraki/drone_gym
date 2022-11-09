# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:35:20 2022

@author: aniaraki
"""
import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
from math import tan, radians
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
from threading import Thread
import threading
from multiprocessing  import Process
from sys import exit
import time


STATES_X=100
STATES_Y=100
STATES_Z=1
init_state=[1,1,1]
### lets define a 1000 m * 250 m = 60 acres world
### lets assume the flight altitude can vary between 60 to 100 m
### The world generates square patches with sizes ranging between (1,10)
WORLD_X=1000
WORLD_Y=250
WORLD_Z_min=60
WORLD_Z_max=100
SEEDS=200
square_size_range=(1,10)
FRAME_W=700
FRAME_H=500
FOV_X=60/2 #degrees for halve of the field of view horizontaly
FOV_Y=47/2 #degrees for halve of the field of view verticaly

class droneEnv():
    
    def __init__(self, name):
        self.name=name
        self.battery=75
        self.main_thread = threading.current_thread()
        if self.name=='cont':
            self.observation_space = Box(low=0, high=255,
                                    shape=(FRAME_H, FRAME_W+1), dtype=np.uint8)
        if self.name=='disc':
### action list for 2d: [0 ,1       ,2    ,3         ,4   ,5        ,6   ,7]
### action list for 2d: [up,up-right,right,right-down,down,down-left,left,left-top ]
            self.action_space=Discrete(8)
        self.location=[100,100,60]
        self.world=self.world_genertor()
### for getting the frame to the agent at all times
        self.thread=Thread(target=self.update_frame, args=(),daemon=True)
        # self.thread.daemon=True
        self.thread.start()
        time.sleep(1)
        print('environment is initialized')        


        
        
    def step(self, action):
### defining navigation #######################################################        
            
         pass
###############################################################################
    
    def reset(self):
        self.world=self.world_genertor()
        self.state=init_state
        self.battery=75
        ###for COARSE Coding there is an auxiliary function to get location from state
        # self.location=self.loc_from_state()
        self.location=[100,100,60]
        self.prev_reward=0
        self.score = 0 
        self.done = False
        x=self.location[0]
        y=self.location[1]
        z=self.location[2]
        frame=self.fetch_frame()
### create observation:
        # observation = [x,y,z,frame] + list(self.prev_actions)
        # observation = np.array(observation)        
       
        
        # Reset shower time
        # self.shower_length = 60 
        return self.state

    

        # return self.state, reward, done, info        
### Auxiliary functions #######################################################        
    def world_genertor(self, seeds=SEEDS, size=(WORLD_Y,WORLD_X)):
         self.world=np.zeros(size, dtype=int)
         square_corners=[]
         for s in range(0,seeds):
             ### corner of each square corner=[x,y]
             corner=[random.randint(0,WORLD_X),random.randint(0,WORLD_Y)]
             ### list of all square corners
             square_corners.append(corner)
             square_size=random.randint(square_size_range[0],square_size_range[1])
             for i in range(0,square_size):
                 for j in range(0,square_size):
                     try:
                         self.world[corner[1]+j][corner[0]+i]=1
                     except:
                         pass
         return self.world
    
    def update_frame (self):
        self.imager_thread_name=threading.current_thread()
        print(self.imager_thread_name)
        while True:
            visible_x=tan(radians(FOV_X))*2*self.location[2]
            visible_y=tan(radians(FOV_Y))*2*self.location[2]
            world_img=np.uint8((1-self.world)*255)
            ### take snap of the sim based on location [x,y,z]
            crop=world_img[int(-visible_y/2+self.location[1]):int(visible_y/2+self.location[1]), int(-visible_x/2+self.location[0]):int(visible_x/2+self.location[0])]
            resized=cv2.resize(crop, (FRAME_W, FRAME_H))
            self.frame=resized

    def fetch_frame(self):
        return self.frame

    def loc_from_state(self):
        state_x_size=(WORLD_X-100-100)/STATES_X
        state_y_size=(WORLD_Y-100-100)/STATES_Y
        state_z_size=(WORLD_Z_max-WORLD_Z_min)/STATES_Z
        loc=[100+state_x_size*(self.state[0]-1), 100+state_y_size*(self.state[1]-1), 60+state_z_size*(self.state[2]-1)]
        ### returns a location=[x,y,z] where states=[1,1,1] corresponds to loc=[100,100,60] meteres
        return loc
    
    def close(self):
        self.imager_thread_name.join()
### method receives frame as np array adds a column the end that represent battery level    
    def concat_battery(self):
        input_frame=self.fetch_frame()
        full_pixels=np.zeros([int(FRAME_H*self.battery/100), 1])
        full_pixels.astype(int)
        empty_pixels=(np.zeros([FRAME_H-int(FRAME_H*self.battery/100),1])+1)*255
        empty_pixels.astype(int)

        # print('len of full pixs: ',str(len(full_pixles())), 'empty pixs: ', str(len(empty_pixels)))
        battery_img=np.uint8(np.concatenate((empty_pixels, full_pixels)))
        cv2.imwrite('justB.png', battery_img)
        self.output_frame=np.concatenate((input_frame, battery_img),axis=1)
        # output_frame=np.append(input_frame,np.zeros([len(input_frame),1]),1)
        print('in shape: ',input_frame.shape, 'out_shape: ', self.output_frame.shape)
        return self.output_frame
        
        
        
        
         
            

env=droneEnv('cont')

W=env.world_genertor()
# cv2.imwrite('withB.png', W)
# 
# frame1=env.fetch_frame()
frame2=env.concat_battery()

# cv2.imwrite('without battery', frame1)
cv2.imwrite('withB.png', frame2)
env.close()
exit()
## closing all the threads except main
# main_thread = threading.current_thread()
# for t in threading.enumerate():
#     if t is env.main_thread:
#         continue
#     print(t)
#     t.join()


### imshow as movie
counter=1
for i in range(0,500):
    try:
        env.location=[100+i,100,60]
        frame1=env.fetch_frame()
        cv2.imshow('drone view', frame1)
        time.sleep(0.1)
        
    except:
        print('frame is not available')
        counter=counter+1
        if counter==20:
            break
       
    if cv2.waitKey(1)==ord('q'):
        # env.close()
        # cam2.capture.release()
        cv2.destroyAllWindows()
        exit(1)
        break        
        
        
        
        
    
