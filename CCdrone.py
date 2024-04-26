# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:50:32 2022

@author: amire


The following script takes in a RECTANGULAR image and generates the complete coverage with defualt:
        1. Front overlap, 2. Side overlap, 3. Drone speed, 4. Wind field and 5. Altitude
"""
from drone_environment import droneEnv
import time
import cv2
import numpy as np
from sys import exit
from configurations import Configs


# wateves
env = droneEnv(observation_mode='cont', action_mode='cont', render=True)

strides_x=int((env.cfg.WORLD_XS[1]-env.cfg.WORLD_XS[0])/env.visible_x)
strides_y=int((env.cfg.WORLD_YS[1]-env.cfg.WORLD_YS[0])/env.visible_y)

step_x=20
step_y=20
LTR=1
steps=0
rewards=[]

for i in range(1):
    # env.close()
    som_obs=env.reset()
    print('Iteration: ', i, '\n supposed location: ', env.location, 'configurations: ', env.cfg.init_location)
    
    while True:
        if LTR==1:
            while env.done==False and abs(env.location[0]-env.cfg.WORLD_XS[1])>1:
                    obs, reward, done, info =env.step([step_x,0,0])
                    # print(f'step: {steps}, reward: {reward}')
                    steps+=1
                    rewards.append(reward)

        if LTR==-1:
            while env.done==False and abs(env.location[0]-env.cfg.WORLD_XS[0])>1:
                    obs, reward, done, info =env.step([step_x,0,0])
                    # print(f'step: {steps}, reward: {reward}')
                    steps+=1
                    rewards.append(reward)      
        step_x=-step_x
        LTR=-LTR
        if env.done==False and abs(env.location[1]-env.cfg.WORLD_YS[1])>1:   
            obs, reward, done, info =env.step([0,step_y,0])
            # print(f'step: {steps}, reward: {reward}')
        else: 
            break
    print(f'length of rewards: {len(rewards)}', f'number of steps: {steps}', 'total rewards: ', sum(rewards))
        




env.close()



