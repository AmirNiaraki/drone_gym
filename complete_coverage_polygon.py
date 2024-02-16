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

# load the polygons from image_editor (should have vertices and edges)
img = np.load("output_polygons.npy")

# Loop through each polygon in the image seperately
# TODO: 
# 	currently resets init position for each polygon. Ideally it should
# 	not teleport from one polygon to the next but instead travel from polygon a
# 	to polygon b.
for polygon in img:

	# set our currently observed polygon
	env.curr_polygon = polygon

	# set the initial drone location within the bounds of the polygon
	# set it within the max of the x and y direction of the polygon
	# TODO:
	#	If we are traveling horizontally we probably just
	# 	need to start from the max x and not care if we are in the max y
	# 	as the algorithm will double back when it reaches the edge in x direction
	#	same is true for vertical travel

	env.location[0] = max(polygon[:, 0].min(), env.cfg.PADDING_X)
	env.location[1] = max(polygon[:, 1].min(), env.cfg.PADDING_Y)
	env.location[2] = 100

	# drone movement within the boundaries of the polygon

if env.orientation == 0:
	LTR = 1
	for i in range(num_iterations):
		som_obs=env.reset()
		print('Iteration: ', i, '\n supposed location: ', env.location, 'configurations: ', env.cfg.init_location)

		while True and not env.done:
			if LTR == 1:
				while not check_boundary(env.location, polygon) and not env.done:
					obs, reward, done, truncs, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0 ,0])
					steps += 1
					rewards.append(reward)

			if LTR == -1:
				while not check_boundary(env.location, polygon) and not env.done:
					obs, reward, done, trunc, info = env.step([-env.visible_x  * (1 - env.cfg.OVERLAP), 0 ,0 ])
					steps += 1
					rewards.append(reward)

			LTR = -LTR

			if not check_boundary(env.location, polygon) and not env.done:
				obs, reward, done, trunc, info = env.step([0, env.visible_y  * (1 - env.cfg.OVERLAP), 0])
			else:
				break

		num_iterations += 1
	env.close()
else:
	UTD = 1  # Up-to-Down
	for i in range(num_iterations):
		som_obs = env.reset()
		print('Iteration: ', i, '\n supposed location: ',
			env.location, 'configurations: ', env.cfg.init_location)

		# Vertical Movement
		while not check_boundary(env.location, polygon) and not env.done:
			if UTD == 1:
				while not check_boundary(env.location, polygon) and not env.done:
					obs, reward, done, truncs, info = env.step(
						[0, env.visible_y * (1 - env.cfg.OVERLAP), 0])
					steps += 1
					rewards.append(reward)

			if UTD == -1:
				while not check_boundary(env.location, polygon) and not env.done:
					obs, reward, done, trunc, info = env.step(
						[0, -env.visible_y * (1 - env.cfg.OVERLAP), 0])
					steps += 1
					rewards.append(reward)

			UTD = -UTD

			obs, reward, done, trunc, info = env.step(
				[env.visible_x * (1 - env.cfg.OVERLAP), 0, 0])

		# Horizontal Movement
		while not check_boundary(env.location, polygon) and not env.done:
			obs, reward, done, truncs, info = env.step(
				[-env.visible_x * (1 - env.cfg.OVERLAP), 0, 0])
			steps += 1
			rewards.append(reward)

		num_iterations += 1

	env.close()

	# Checks if we are within the boundary of the polygon
	def check_boundary(location, polygon):
		x_min, x_max = polygon[:, 0].min(), polygon[:, 0].max()
		y_min, y_max = polygon[:, 1].min(), polygon[:, 1].max()

		return x_min <= location[0] <= x_max and y_min <= location[1] <= y_max