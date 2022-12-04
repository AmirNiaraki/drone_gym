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
init_location=[100,100,60]
### lets define a 1000 m * 250 m = 60 acres world
### lets assume the flight altitude can vary between 60 to 100 m
### The world generates square patches with sizes ranging between (1,10)
WORLD_XS=[100,1900]
WORLD_YS=[100,900]
WORLD_ZS=[60,100]

SEEDS=200
square_size_range=(1,10)
FRAME_W=350
FRAME_H=250
FOV_X=60/2 #degrees for halve of the field of view horizontaly
FOV_Y=47/2 #degrees for halve of the field of view verticaly
FULL_BATTERY=100
MAX_SPEED=20 
PADDING=100
FPS=10
### the padded area of the world is were the drone cannot go to but may appear in the frame

class droneEnv(gym.Env):
    
    def __init__(self, name):
        super(droneEnv, self).__init__()
        
        self.name=name
        self.location=init_location
        self.world=self.world_genertor()
        self.battery=FULL_BATTERY # [x,y,z,] m
        self.done=False
        self.reward=0
        self.total_reward=0
        self.step_count=0
        
        if self.name=='cont':
            self.observation_space = Box(low=0, high=255,
                                    shape=(FRAME_H, FRAME_W+1), dtype=np.uint8)
            self.action_space=Box(low=-MAX_SPEED, high=MAX_SPEED, shape=(3,), dtype=np.float32)
        if self.name=='disc':
### action list for 2d: [0 ,1       ,2    ,3         ,4   ,5        ,6   ,7]
### action list for 2d: [up,up-right,right,right-down,down,down-left,left,left-top ]
            self.action_space=Discrete(8)
        
        


### for getting the frame to the agent at all times
        self.thread=Thread(target=self.update_frame, args=(),daemon=True)
        self.thread.start()
        time.sleep(1)

        # self.reset()
        
        print('environment is initialized')        


        
        
    def step(self, action):
### defining navigation #######################################################     
### let's assume each step takes 1 second and moves the agent for =1 (s) * V (m/s)    
         # self.prev_actions.append(action)
        action=np.array(action)/FPS

        # for i in range (1,FPS+1):
            
        #     if action[0]<0:
        #         self.location[0]=max(self.location[0]+action[0], WORLD_XS[0])  
        #     else:
        #         self.location[0]=min(self.location[0]+action[0], WORLD_XS[1])
        #     if action[1]<0:
        #         self.location[1]=max(self.location[1]+action[1], WORLD_YS[0])  
        #     else:
        #         self.location[1]=min(self.location[1]+action[1], WORLD_YS[1])
        #     if action[2]<0:
        #         self.location[2]=max(self.location[2]+action[2], WORLD_ZS[0])  
        #     else:
        #         self.location[2]=min(self.location[2]+action[2], WORLD_ZS[1])
             
        self.reward=-self.move_cost()
         
        if self.battery<1:
             self.reward-=10
             self.done=True
             env.close()
         
        observation=self.fetch_frame()
        
            
        self.reward+=self.fetch_anomaly()
        self.total_reward+=self.reward
        self.step_count+=1
        # if self.fetch_anomaly()>0:
        #     print('step',self.step_count, '\n this reward: ', self.reward, '\n')
        #     print('Total rewards is:', self.total_reward)
        self.reward.astype(np.float32)
        info={}

        # self.steps_taken=
        
        # self.reward=
        
        
        return observation, self.reward, self.done, info
 
         
         
         
###############################################################################
    
    def reset(self):
        self.location=init_location
        self.world=self.world_genertor()
        self.battery=FULL_BATTERY # [x,y,z,] m
        self.reward=0
        self.total_reward=0
        self.step_count=0
        ###for COARSE Coding there is an auxiliary function to get location from state
        self.prev_reward=0
        self.score = 0 
        self.done = False

        observation=self.fetch_frame()

        return observation

    

        # return self.state, reward, done, info        
### Auxiliary functions #######################################################        
    def world_genertor(self, seeds=SEEDS, size=(WORLD_YS[1]+PADDING,WORLD_XS[1]+PADDING)):
         self.world=np.zeros(size, dtype=int)
         square_corners=[]
         for s in range(0,seeds):
             ### corner of each square corner=[x,y]
             corner=[random.randint(PADDING,WORLD_XS[1]),random.randint(PADDING,WORLD_YS[1])]
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
        
        while True:
            self.visible_x=tan(radians(FOV_X))*2*self.location[2]
            self.visible_y=tan(radians(FOV_Y))*2*self.location[2]
            self.world_img=np.uint8((1-self.world)*255)
            ### take snap of the sim based on location [x,y,z]
            ### visible corners of FOV in the form boundaries= [y,y+frame_h,x,x+frame_w]
            self.boundaries=[int(-self.visible_y/2+self.location[1]),int(self.visible_y/2+self.location[1]), int(-self.visible_x/2+self.location[0]),int(self.visible_x/2+self.location[0])]
            crop=self.world_img[self.boundaries[0]:self.boundaries[1],self.boundaries[2]:self.boundaries[3]]
            resized=cv2.resize(crop, (FRAME_W, FRAME_H))
            added_battery=self.concat_battery(resized)
            self.frame=added_battery
            if self.done==True:
                break
                
        print('Frame Update stopping...  ',  self.imager_thread_name)


    def fetch_frame(self):
        
        return self.frame
    
    
    def fetch_anomaly(self):
        observation=self.fetch_frame()
        nobat=observation[0:FRAME_H,0:FRAME_W]
        
        score=FRAME_H*FRAME_W-np.sum(nobat/255, dtype=np.int32)
        
        # self.world[int(-self.visible_y/2+self.location[1]):int(self.visible_y/2+self.location[1]), int(-self.visible_x/2+self.location[0]):int(self.visible_x/2+self.location[0])]=0
        return score

        
        

    def loc_from_state(self):
        state_x_size=(WORLD_XS[1]-WORLD_XS[0])/STATES_X
        state_y_size=(WORLD_YS[1]-WORLD_YS[0])/STATES_Y
        state_z_size=(WORLD_YS[1]-WORLD_YS[0])/STATES_Z
        loc=[WORLD_XS[0]+state_x_size*(self.state[0]-1), WORLD_YS[0]+state_y_size*(self.state[1]-1), WORLD_ZS[0]+state_z_size*(self.state[2]-1)]
        ### returns a location=[x,y,z] where states=[1,1,1] corresponds to loc=[100,100,60] meteres
        return loc
    
    def close(self):
        self.done=True
        time.sleep(0.1)
        self.imager_thread_name.join()
### method receives frame as np array adds a column the end that represent battery level    
    def concat_battery(self, input_frame):
        full_pixels=np.zeros([int(FRAME_H*self.battery/100), 1])
        full_pixels.astype(int)
        empty_pixels=(np.zeros([FRAME_H-int(FRAME_H*self.battery/100),1])+1)*255
        empty_pixels.astype(int)

        # print('len of full pixs: ',str(len(full_pixles())), 'empty pixs: ', str(len(empty_pixels)))
        battery_img=np.uint8(np.concatenate((empty_pixels, full_pixels)))
        cv2.imwrite('justB.png', battery_img)
        self.output_frame=np.concatenate((input_frame, battery_img),axis=1)
        # output_frame=np.append(input_frame,np.zeros([len(input_frame),1]),1)
        return self.output_frame
        
    def move_cost(self):
### method to find the step cost based on drag force, for now everything costs 1
        self.cost=1
        self.battery=self.battery-1
        return self.cost
        
        
         
            

env=droneEnv('cont')
# obs=env.step([10,5,1])
# cv2.imshow('drone view', obs)
# if cv2.waitKey(1)==ord('q'):
#     env.close()
#     cv2.destroyAllWindows()

# exit()
# W=env.world_genertor()
# cv2.imwrite('withB.png', W)
# obs=env.step([500,100,10])
# print(obs)
# cv2.imwrite('observation', obs)

# frame1=env.fetch_frame()
# frame2=env.concat_battery()

# cv2.imwrite('without battery', frame1)
# cv2.imwrite('withB.png', frame2)
# env.close()

## closing all the threads except main
# main_thread = threading.current_thread()
# for t in threading.enumerate():
#     if t is env.main_thread:
#         continue
#     print(t)
#     t.join()

# exit()
### imshow as movie
counter=1
obs=env.step([10,5,0])
for i in range(0,1000):
    # obs=env.step([10,5,0])
    try:
        # env.location=[100+i,100,60]

        # obs=env.step([10,5,0])
        world_img=env.world_img
        
        frame1=env.fetch_frame()
        # cv2.imshow('drone view', frame1)
        gray_BGR = cv2.cvtColor(world_img, cv2.COLOR_GRAY2BGR)
        # print(env.boundaries[0],env.boundaries[2],env.boundaries[1],env.boundaries[1])
        img=cv2.rectangle(gray_BGR, (env.boundaries[2],env.boundaries[0]),(env.boundaries[3],env.boundaries[1]),(255, 0, 0),5)

        # cv2.imshow('World view', img)
        cv2.imshow('drone view', frame1)
        time.sleep(1/FPS)
        
    except:
        # print('frame is not available')
        counter=counter+1
        if counter==2000:
            break
       
    if cv2.waitKey(1)==ord('q'):
        env.close()
        # cam2.capture.release()
        cv2.destroyAllWindows()
        exit(1)
        break        
        
        
        
        
    
