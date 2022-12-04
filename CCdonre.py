# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:50:32 2022

@author: amire
"""
from IPP_drone_path_planner import droneEnv
import time
import cv2
import numpy as np
from sys import exit
from configurations import Configs

env = droneEnv('cont', render=True)

strides_x=int((env.cfg.WORLD_XS[1]-env.cfg.WORLD_XS[0])/env.visible_x)
strides_y=int((env.cfg.WORLD_YS[1]-env.cfg.WORLD_YS[0])/env.visible_y)

step_x=env.visible_x
step_y=env.visible_y
LTR=1
steps=0
rewards=[]

for i in range(strides_y):
    if LTR==1:
        for j in range(strides_x):
            
            if env.done==False:
                obs, reward, done, info =env.step([step_x,0,0])
                steps+=1
                rewards.append(reward)
            if env.done==True:
                break
    if LTR==-1:

        for j in reversed(range(strides_x)):
            
            if env.done==False:
                obs, reward, done, info =env.step([step_x,0,0])
                steps+=1
                rewards.append(reward)
            if env.done==True:
                break        
    
    if env.done==True:
        break
        

    step_x=-step_x
    LTR=-LTR    
    obs, reward, done, infor =env.step([0,step_y,0])

    
env.close()



