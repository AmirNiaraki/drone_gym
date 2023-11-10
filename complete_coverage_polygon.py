from drone_environment import droneEnv
import time
import cv2
import numpy as np
from sys import exit
from configurations import Configs
from math import tan, radians, degrees, acos, sqrt
from PIL import Image
from sys import exit

env = droneEnv('cont', render=True)

# step size should be calculated based on a target overlap % (75%)
LTR 	= 1 	# Left-to-Right
steps 	= 0
num_iterations = 1
rewards = []

# get range of coverage from list of polygon points
img_points = np.load("output_image_points.npy")
area_bounds =  {"minx" : np.min(img_points[:,1]), 
				"maxx" : np.max(img_points[:,1]),
				"miny" : np.min(img_points[:,0]),
				"maxy" : np.max(img_points[:,0])}

print("Area to be searched:\n" + 
	  "x-range: " + str(area_bounds["minx"]) + "-" + str(area_bounds["maxx"]) + "\n" +
	  "y-range: " + str(area_bounds["miny"]) + "-" + str(area_bounds["maxy"]) + "\n")


som_obs=env.reset()

# initialize drone's starting position
env.world = np.load("output_image.npy")
env.location[0] = max(area_bounds["minx"], env.cfg.PADDING_X)
env.location[1] = max(area_bounds["miny"], env.cfg.PADDING_Y)
env.location[2] = 60

# print("res:" + str(env.world.shape))

# while the environment isn't finished
while not env.done:
	# go right
	while LTR == 1 and env.location[0] < area_bounds["maxx"] and env.location[0] < env.cfg.WORLD_XS[1]:
		# take step
		obs, reward, done, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0 ,0])
		steps += 1
		rewards.append(reward)

	# go left
	while LTR == -1 and env.location[0] > area_bounds["minx"] and env.location[0] > env.cfg.WORLD_XS[0]:
		# take step left
		obs, reward, done, info = env.step([-env.visible_x * (1 - env.cfg.OVERLAP), 0 ,0])
		steps += 1
		rewards.append(reward)

	LTR = -LTR

	# drone should be at left/right edge of bounded area
	if (env.location[1] < area_bounds["maxy"]):
		obs, reward, done, info = env.step([0, env.visible_y  * (1 - env.cfg.OVERLAP), 0])
	# if the drone can't move down it must be finished
	else:
		break

	num_iterations += 1
env.close()