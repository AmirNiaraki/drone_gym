# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:50:32 2022

@author: amire


The following script takes in a RECTANGULAR image and generates the complete coverage with defualt:
	1. Front Overlap
	2. Side Overlap
	3. Drone Speed
	4. Wind Field
	5. Altitude


Serves as benchmark for other algorithms to beat.

TO IMPROVE:
	Ideally, the drone recieves a list of "keypoints" or coordinates to hit to accomplish complete coverage, taking into account front & side overlap.
	The % overlap is a function of x_step, y_step.
	To optimize, the drone's path should have the minimum number of turns. For example, it's best to be parallel to the longest edge.
	TODO: Research how to generate optimized lawn-mower pathing.
	TODO: Generate this optimized list of keypoints to cover a rectangle. Eventually a polygon.
	TODO: Figure out the necessary padding.
"""
from drone_environment import droneEnv
import time
import cv2
import numpy as np
from sys import exit
from configurations import Configs
from math import tan, radians, degrees, acos, sqrt

env = droneEnv(render=True)

# Number of strides necessary to cover rectangle (?)
strides_x = int((env.cfg.WORLD_XS[1] - env.cfg.WORLD_XS[0]) / env.visible_x)
strides_y = int((env.cfg.WORLD_YS[1] - env.cfg.WORLD_YS[0]) / env.visible_y)

# step size should be calculated based on a target overlap % (75%)
LTR 	= 1 	# Left-to-Right
steps 	= 0
num_iterations = 1
rewards = []

for i in range(num_iterations):
	som_obs=env.reset()
	print('Iteration: ', i, '\n supposed location: ', env.location, 'configurations: ', env.cfg.init_location)

	while True and not env.done:
		if LTR == 1:
			while abs(env.location[0] - env.cfg.WORLD_XS[1]) > 1 and not env.done:
				obs, reward, done, trunc, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0 ,0])
				steps += 1
				rewards.append(reward)

		if LTR == -1:
			while abs(env.location[0] - env.cfg.WORLD_XS[0]) > 1 and not env.done:
				obs, reward, done, trunc, info = env.step([-env.visible_x  * (1 - env.cfg.OVERLAP), 0 ,0 ])
				steps += 1
				rewards.append(reward)

		LTR = -LTR

		if abs(env.location[1] - env.cfg.WORLD_YS[1]) > 1 and not env.done:
			obs, reward, done, trunc, info = env.step([0, env.visible_y  * (1 - env.cfg.OVERLAP), 0])
		else:
			break

	num_iterations += 1
env.close()