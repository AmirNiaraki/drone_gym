# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:38:41 2022

@author: amireniaraki
"""
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

step_x=25 
step_y=20
LTR=1
steps=0
rewards=[]



while True:

    if LTR==1:
        while env.done==False and abs(env.location[0]-env.cfg.WORLD_XS[1])>1:
            
                obs, reward, done, info =env.step([step_x,0,0])
                steps+=1
                rewards.append(reward)
            
    if LTR==-1:

        while env.done==False and abs(env.location[0]-env.cfg.WORLD_XS[0])>1:
                obs, reward, done, info =env.step([step_x,0,0])
                steps+=1
                rewards.append(reward)      
    
        
    
    step_x=-step_x
    LTR=-LTR
    
    if env.done==False and abs(env.location[1]-env.cfg.WORLD_YS[1])>1:   
        obs, reward, done, infor =env.step([0,step_y,0])
    else: 
        break

    
env.close()



