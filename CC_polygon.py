from drone_environment import droneEnv
import numpy as np
import argparse
from world_generator import WorldGenerator
from process_image import process

class Complete_Coverage():
	def __init__(self, args):

		wname = None
		gen = True

		if args.filename:
			self.world_name = args.filename
			if '.npy' not in args.filename:
				process(args.filename, 1)
				wname = args.filename.rsplit(".", 1)[0] + ".npy"
		if args.randomSetFilename:
			WorldGenerator(args.randomSetFilename)
			wname = args.randomSetFilename
		if args.random:
			WorldGenerator(file_name='random_world')
			wname = 'random_world.npy'
		if args.randomSetSize:
			x, y = args.randomSetSize
			WorldGenerator(file_name='random_world', world_y=y, world_x=x)
			wname = 'random_world.npy'
		if args.randomSetSizeFilename:
			wname, x, y = args.randomSetSizeFilename
			WorldGenerator(file_name=wname, world_y=y, world_x=x)

		env = droneEnv(render=True, generate_world=gen, wname=wname)

		self.search(env)

	def findMinMax(self, shape):
		
		area_bounds =  {"miny" : np.min(shape[:,1]), 
				"maxy" : np.max(shape[:,1]),
				"minx" : np.min(shape[:,0]),
				"maxx" : np.max(shape[:,0])}		

		return area_bounds			

	# assumes we always start in the upper left corner
	# find the top most (y) point and then find an x within the shape bounds
	def findStart(self, shape):
		# find the highest point in the y direction (min most)
		topmost = min(shape, key=lambda point : point[1])

		x, y = topmost

		# Find the bounds of the shape in the x-direction
		minx = min(point[0] for point in shape)
		maxx = max(point[0] for point in shape)

		if x < minx:
			x = minx
		elif x > maxx:
			x = maxx

		return x, y

	def getLines(self, shape):
		lines = []

		xsorted = []
		ysorted = []

		# for each point pair in shape given
		for i in range(len(shape) - 1):
			# pair up points to form a line
			line = np.array([shape[i], shape[i + 1]])
			# add line to list of lines
			lines.append(line)

			# add line sorted by lowest x value 
			xsorted.append(line[np.argsort(line[:, 0])])
			# add line sorted by lowest y value
			ysorted.append(line[np.argsort(line[:, 1])])

		# add the closing line segment from last point to the first point of the shape
		lines.append(np.array([shape[-1], shape[0]]))
		#sort the closing line segment by x and y coordinate and add
		xsorted.append(lines[-1][np.argsort(lines[-1][:, 0])])
		ysorted.append(lines[-1][np.argsort(lines[-1][:, 1])])

		return xsorted, ysorted
	
	def binarySearch(self, lines, val, coordinate='x'):
		low = 0
		high = len(lines) - 1
		min_line = None
		max_line = None

		while low <= high:
			mid = (low + high) // 2
			mid_val = lines[mid][:, 0 if coordinate == 'x' else 1].mean()

			if mid_val < val:
				low = mid + 1
				min_line = lines[mid]
			elif mid_val > val:
				high = mid - 1
				max_line = lines[mid]
			else:
				return lines[mid], None

		return max_line, min_line

	# get upper and lower bounds of the shape given 
	# lines: numpy array list of lines composing a shape
	# val: 
	def getBounds(self, lines, val, coordinate='x'):
		max, min = self.binarySearch(lines, val, coordinate)

		if max is None or min is None:
			return None

		max_x1, max_y1 = max[0]
		max_x2, max_y2 = max[1]
		min_x1, min_y1 = min[0]
		min_x2, min_y2 = min[1]

		# calc the slope (m) and intercept (b) for each line
		max_m = (max_y2-max_y1)/float((max_x2-max_x1))
		max_b = max_y1 - max_m * max_x1

		min_m = (min_y2-min_y1)/float((min_x2-min_x1))
		min_b = min_y1 - min_m * min_x1

		if coordinate == 'x':
			upper = (val - max_b) / max_m
			lower = (val - min_b) / min_m

			return upper, lower
		else: # y-coordinate
			upper = max_m * val + max_b
			lower = min_m * val + min_b

			return upper, lower
	
	def search(self, env):

		steps 	= 0
		num_iterations = 1
		rewards = []

		env.world = np.load(env.world_name)
		wname = env.world_name.rsplit(".", 1)[0]
		env.points = np.load(wname + "_points.npy")
		print(env.points)

		for shape in env.points:
			print(shape)
			area_bounds = self.findMinMax(shape)

			# set starting location top/left most point within the shape
			minx, miny = self.findStart(shape)

			env.location[0] = max(minx, env.cfg.PADDING_X)
			env.location[1] = max(miny, env.cfg.PADDING_Y)

			# get lines sorted by x or y ranges
			xsorted, ysorted = self.getLines(shape)

			# set orientation for shape
			if (area_bounds["maxx"] - area_bounds["minx"]) > (area_bounds["maxy"] - area_bounds["miny"]):
				# wider than tall --> LTR
				env.orientation = 0

				obs, reward, done, trunc, info = self.horizontalSearch(env, xsorted, num_iterations, rewards)
			else:
				# taller than wider --> UTD
				env.orientation = 1
				obs, reward, done, trunc, info = self.verticalSearch(env, ysorted, num_iterations, rewards)

			# TODO: after shape loop, we need to connect this to the next shape somehow

		env.close()

	def horizontalSearch(self, env, lines, num_iterations, rewards):
		# left-to-right: LTR: (->) & -LTR: (<-)
		LTR = 1

		for i in range(num_iterations):
			env.reset()
			
			while True and not env.done:
				if LTR == 1:
					# get the bound for shape in +x direction
					# y is static, use it to solve for intersection to find x border
					bound, _ = self.getBounds(lines, env.location[1], 'x')

					# while current location is within bounds
					while abs(env.location[0] - (bound + env.cfg.WORLD_XS[0])) > 1 and not env.done:
						obs, reward, done, truncs, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0 ,0])
						steps += 1
						rewards.append(reward)

				if LTR == -1:
					# get the bounds for shape in -x direction
					_, bound = self.getBounds(lines, env.location[0], 'x')

					while abs(env.location[0] + (bound - env.cfg.WORLD_XS[0])) > 1 and not env.done:
						obs, reward, done, trunc, info = env.step([-env.visible_x  * (1 - env.cfg.OVERLAP), 0 ,0 ])
						steps += 1
						rewards.append(reward)

				LTR = -LTR

				# TODO: Fix this
				if abs(env.location[1] - env.cfg.WORLD_YS[1]) > 1 and not env.done:
					obs, reward, done, trunc, info = env.step([0, env.visible_y  * (1 - env.cfg.OVERLAP), 0])
				else:
					break

			num_iterations += 1

		return obs, reward, done, trunc, info
	
	def verticalSearch(self, env, lines, num_iterations, rewards):
		# Up-to-Down: UTD: (^) & -UTD: (⌄)
		UTD = 1

		for i in range(num_iterations):
			env.reset()

			# Vertical Movement
			while True and not env.done:
				if UTD == 1:

					bound, _ = self.getBounds(lines, env.location[0], 'y')

					while abs(env.location[1] - (bound + env.cfg.WORLD_YS[0])) > 1 and not env.done:
						obs, reward, done, truncs, info = env.step([0, env.visible_y * (1 - env.cfg.OVERLAP), 0])
						steps += 1
						rewards.append(reward)

				if UTD == -1:

					_, bound = self.getBounds(lines, env.location[0], 'y')

					while abs(env.location[1] - (bound + env.cfg.WORLD_YS[0])) > 1 and not env.done:
						obs, reward, done, trunc, info = env.step([0, -env.visible_y * (1 - env.cfg.OVERLAP), 0])
						steps += 1
						rewards.append(reward)

				UTD = -UTD

			# Horizontal Movement
			if abs(env.location[0] - env.cfg.WORLD_XS[1]) > 1 and not env.done:
				obs, reward, done, truncs, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0, 0])

			num_iterations += 1

		return obs, reward, done, truncs, info

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run complete coverage against a search area.')
	parser.add_argument('-f', '--filename', type=str, help='Specify the filename: Non-.npy files will be processed before running')
	parser.add_argument('-fr', '--randomSetFilename', type=str, help='specify a filename to give a randomly generated world.')
	parser.add_argument('-r', '--random', action='store_true', help='Randomly generate a world to search.')
	parser.add_argument('-rs', '--randomSetSize', nargs=2, help='Randomly generate a world of a specified size.')
	parser.add_argument('-frs', '--randomSetSizeFilename', nargs=3, help='Specify a filename for a randomly generate world of a set size')
	# parser.add_argument('-w', '--wind', type=float, help='Set wind speed.')
	# parser.add_argument('-b', '--battery', action='store_true', help='Turn off battery for testing.')

	args = parser.parse_args()

	Complete_Coverage(args)