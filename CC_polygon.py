from drone_environment import droneEnv
import numpy as np
import argparse
import math
from world_generator import WorldGenerator
from process_image import process

class Complete_Coverage():
	def __init__(self, args):

		wname, starts = None, None
		x, y = None, None

		# parse the flags given to setup a known world or generate one at random
		wname, x, y = self.parseArgs(args)

		# set the initial position
		starts = self.setStart(wname)

		# init environment
		env = droneEnv(render=True, generate_world=False, wname=wname, CC=starts)

		# if we are setting a specific size of map, override cfg's world sizes
		if x is not None and y is not None:
			env.cfg.WORLD_XS = (env.cfg.PADDING_X, int(x))
			env.cfg.WORLD_YS = (env.cfg.PADDING_Y, int(y))

		# load the world from file
		env.world = np.load(env.world_name)

		# toggle flags
		if args.togglePath:
			# show path of drone
			env.show_path = True
			# save the path as a file
			env.save_path = True
		if args.toggleBounds:
			# show bounds of shape(s) of search
			env.show_bounds = True
		if args.toggleBattery:
			# turn off terminating case when battery < 0 (for testing)
			env.t_bat = False
		if args.maxWind:
			# set the max wind in x and y direction
			env.cfg.DEFAULT_WIND = args.maxWind
		if args.test:
			# run toggles in test mode (path, bounds, battery off, and save path)
			env.show_path = True
			env.show_bounds = True
			env.t_bat = False
			env.save_path = True
		
		# remove all padding and optimize the order of shapes
		self.sanitize(env)

		# run search
		self.search(env)

	# Handle flag logic from command line
	def parseArgs(self, args):
		x = None
		y = None
		wname = None

		# if a desired size is set
		if args.setSize:
			x, y = args.setSize
			x = int(x)
			y = int(y)

		# if given a filename argument
		if args.filename:
			wname = args.filename
		# if random world is requested
		if args.random:
			# randomly gen world with file name
			WorldGenerator().run(file_name = wname.rsplit(".", 1)[0] if wname is not None else None, world_y=y, world_x=x)
			# set name as random_world.npy if none is provided, else set as provided name	
			wname = 'random_world.npy' if wname is None else (wname + ".npy" if "." not in wname else wname.rsplit(".", 1)[0] + ".npy")

		# if image processing flag is present
		if args.process:
				# find the base filename w/o extension
				file = wname.rsplit(".", 1) if wname is not None else None

				# if file has no extension or is None, randomly generate
				if file is None or len(file) == 1:
					# do not attempt to load from file, randomly generate with root filename
					wname = WorldGenerator().run(file_name=wname, world_y=y, world_x=x)
					# reset file to the new wname given by worldGenerator
					file = wname.rsplit(".", 1)

				# process the loaded world or randomly generated world
				wname = process(wname, 1) if file[1] != "npy" else process(file[0] + ".png", 1)

		# if no flags were given, randomly generate at default size w/o processing
		if not (args.filename or args.random or args.process):
			WorldGenerator().run(file_name=wname, world_y=y, world_x=x)

			wname = 'random_world.npy'

		return wname, x, y

	# remove all padding from the list and optimize the order of shapes
	def sanitize(self, env):
		# Get the filename without extension
		wname = env.world_name.rsplit(".", 1)[0]
		# Get the list of points from the file
		env.points = np.load(wname + "_points.npy")

		# Initialize a list to store the cleaned points
		cleaned_points = []

		# Iterate over each shape in the points array
		for shape in env.points:
			# Find the index of the first occurrence of padding points (-1, -1) in the shape
			first_padding_index = np.argmax(np.all(shape == (-1, -1), axis=1))
			if first_padding_index == 0:
				# If no padding points are found, set the index to the length of the shape
				first_padding_index = len(shape)
			# Remove padding points from the shape and append
			cleaned_points.append(shape[:first_padding_index])

		# Convert the list of cleaned points back to a numpy array
		env.points = cleaned_points

		# Optimize the order of shapes in the list to create the shortest path between each shape
		env.points = self.optimizeShapes(env.points)

	# optimize the order of shapes in the list to always travel to the nearest shape
	# from the one nearest to the origin
	def optimizeShapes(self, shapes):
		# reordered shape list
		reordered = []
		visited = set()

		min_dist = float('inf')
		start = None

		# find the shape closest to the origin (0,0)
		for i, shape in enumerate(shapes):
			dist = np.linalg.norm(np.mean(shape, axis=0))
			if dist < min_dist:
				min_dist = dist
				start = i
		
		curr = start

		# reorder the shapes starting from the closest to the origin (0, 0)
		while len(reordered) < len(shapes):
			reordered.append(shapes[curr])
			visited.add(curr)

			min_dist = float('inf')
			next = None

			for i, shape in enumerate(shapes):
				if i not in visited:
					dist = np.linalg.norm(np.mean(shapes[curr], axis=0) - np.mean(shape, axis=0))
					if dist < min_dist:
						min_dist = dist
						next = i

			curr = next

		return reordered

	# find the optimal start point for a given shape
	def findStart(self, lines, orient):

		min_point = None

		bias = orient ^ 1

		# Filter out lines with (-1, -1)
		valid_lines = [line for line in lines if not np.any(line == [-1, -1])]

		# Create a generator expression to iterate over valid lines and compute necessary values
		generator_expr = ((line[np.argmin(line[:, bias])], np.min(line[:, bias]), np.min(line[:, 1 - 1])) for line in valid_lines)

		# Find the point with the minimum values (if tied, compare the other cordinates in the pairs)
		if orient:
			min_point = min(generator_expr, key=lambda x: (x[1], x[2], x[0][0]))
		else:
			min_point = min(generator_expr, key=lambda x: (x[1], x[2], x[0][1]))

		# Extract the point pair from the result
		absolute_min_point = min_point[0]
		
		return absolute_min_point
	
	def findEnd(self, shape, cord):
		return max(shape, key=lambda point: point[cord ^ 1])

	# get and set the initial start values
	def setStart(self, wname):
		sorted = []
		#1. load npy array from points, reorganize to optimize search order
		points = self.optimizeShapes(np.load(wname.rsplit(".", 1)[0] + "_points.npy"))
		#2. get orientation for first shape of points
		orient = self.getOrientation(self.findMinMax(points[0]))
		#3. run self.getLines on first shape
		sorted = self.getLines(points[0]) 
		#4. get the starting points for the first shape
		start = self.findStart(sorted[orient], orient)

		return start

	# find the bounds of the given shape in the x and y directions
	def findMinMax(self, shape):
		
		area_bounds =  {"miny" : np.min(shape[:,1]), 
				"maxy" : np.max(shape[:,1]),
				"minx" : np.min(shape[:,0]),
				"maxx" : np.max(shape[:,0])}		

		return area_bounds			

	# take a list of points in a shape (x,y pairs) and return a two sorted lists of lines:
	def getLines(self, shape):
		lines = []

		xsorted = []
		ysorted = []

		# sorter for lines to allow comparison of min coordinate value
		# if min value  is equal in both lines, prioritize line with smaller range
		def sort_key(line, cord=0):
			start = min(line[0][cord], line[1][cord])
			end   = max(line[0][cord], line[1][cord])

			return (start, end)

		# for each point pair in shape given
		for i in range(len(shape) - 1):
			# pair up points to form a line
			line = np.array([shape[i], shape[i + 1]])
			# add line to list of lines
			lines.append(line)

		# sort lines by x range of point pairs that cmopose a line decending order
		xsorted = sorted(lines, key=lambda line: sort_key(line, cord=0))
		# sort lines by y range of point pairs that cmopose a line decending order
		ysorted = sorted(lines, key=lambda line: sort_key(line, cord=1))

		return xsorted, ysorted
	
	# search helper to find the min and max lines for a given direction of travel
	# based on a constant value of either x or y
	def binarySearch(self, lines, val, cord):
		# Initialize variables
		curr = 0
		end = len(lines) - 1
		first_line = None
		last_line = None

		# iterate through the list of lines
		while curr <= end:
			# Calculate midpoint
			mid = curr + (end - curr) // 2
			
			# Calculate min and max values for the current midpoint
			line_min = min(lines[mid][0][cord], lines[mid][1][cord])
			line_max = max(lines[mid][0][cord], lines[mid][1][cord])

			# Check if the value falls within the range of the current line
			if line_min <= val <= line_max:
				# Set first and last line to current midpoint
				first_line = mid
				last_line = mid

				# Search for the first occurrence on the left
				left = mid - 1
				while left >= curr:
					if lines[left][0][cord] == lines[left][1][cord]:
						left -= 1
						continue
					if self.checkValue(lines[left], val, cord):
						first_line = left
					left -= 1

				# Search for the last occurrence on the right
				right = mid + 1
				while right <= end:
					if lines[right][0][cord] == lines[right][1][cord]:
						right += 1
						continue
					if self.checkValue(lines[right], val, cord):
						last_line = right
					right += 1

				# Break out of the loop since the range is found
				break
			# Adjust search boundaries based on comparison with min value
			elif val < line_min:
				end = mid - 1
			else:
				curr = mid + 1

		# Return the lines corresponding to the found range
		return lines[first_line], lines[last_line]

	# check if val is within line
	def checkValue(self, line, val, cord):
		line_min = min(line[0][cord], line[1][cord])
		line_max = max(line[0][cord], line[1][cord])

		return line_min <= val <= line_max
	
	# find the complimentary x or y coordinate along two bounding lines given an x or y constant
	def getBounds(self, lines, val, cord):
		upper_bound = None
		lower_bound = None

		upper, lower = self.binarySearch(lines, val, cord)

		# if there aren't two lines return Exception (error in code)
		if upper is None or lower is None:
			raise Exception("Upper/Lower bound cannot be None type")

		# find the upper/lower of both x and y for both end points in both lines
		upper_x1, upper_y1 = upper[0]
		upper_x2, upper_y2 = upper[1]
		# lower being the lower bound line
		lower_x1, lower_y1 = lower[0]
		lower_x2, lower_y2 = lower[1]

		# check slope for vertical/horizontal line
		if upper_y2 == upper_y1 or upper_x2 == upper_x1:
			upper_bound = upper_x1 if cord else upper_y1

		if lower_y2 == lower_y1 or lower_x2 == lower_x1:
			lower_bound = lower_x1 if cord else lower_y1

		if upper_bound is None:
			upper_m = (upper_y2 - upper_y1) / float((upper_x2 - upper_x1))
			upper_b = upper_y1 - upper_m * upper_x1
			
			if cord:
				upper_bound = (val - upper_b) / upper_m
			else:
				upper_bound = (upper_m * val) + upper_b
		if lower_bound is None:
			lower_m = (lower_y2 - lower_y1) / float((lower_x2 - lower_x1))
			lower_b = lower_y1 - (lower_m * lower_x1)
			
			if cord:
				lower_bound = (val - lower_b) / lower_m
			else:
				lower_bound = (lower_m * val) + lower_b
				
		return math.ceil(upper_bound), math.ceil(lower_bound)

	# find the orientation based on the x length and y length of the shape.
    # if x < y; orientation is vertical (1). Else, orientation is horizontal (0)
	def getOrientation(self, area_bounds):
		if (area_bounds["maxx"] - area_bounds["minx"]) < (area_bounds["maxy"] - area_bounds["miny"]):
			return 1
		else:
			return 0

	# search algorithm to exhaustively search a series of shapes within an area varying by change in orientation
	def search(self, env):

		tot_steps 	= 0
		tot_num_iterations = 1
		tot_rewards = []

		sorted = []
		tot_obs = []
		tot_info = []
		end = None

		# for each shape within the area, run a version of complete coverage to search
		for shape in env.points:
			# find the bounds of the shape as if it was a box
			env.area_bounds = self.findMinMax(shape)

			# get lines sorted by x or y ranges
			# sorted[0] = sorted by x range, sorted[1] = sorted by y range
			sorted = self.getLines(shape)

			# 0 means LTR, 1 means UTD
			env.orientation = self.getOrientation(env.area_bounds)

			env.end = self.findEnd(shape, env.orientation)

			# sets start x to the first index of the sorted xsorted line list's x value
			start = self.findStart(sorted[1], env.orientation)

			# if we have seached at least one shape, move to our next shape via step
			# to record costs of moving between search areas
			if end is not None:
				self.moveToNewShape(env, start)
			env.location[0] = max(start[0], env.cfg.PADDING_X)
			# sets start y to the first index of the sorted xsorted line list's y value
			env.location[1] = max(start[1], env.cfg.PADDING_Y)

			# if env.orientation == 1, vertical search
			if env.orientation:
				obs, reward, done, trunc, info, num_iterations, steps = self.verticalSearch(env, sorted, tot_steps, tot_num_iterations, tot_rewards)
				# record shape's search outputs as a running total
				tot_obs.append(obs)
				tot_info.append(info)
				tot_rewards.append(reward)
				tot_num_iterations += num_iterations
				tot_steps += steps
			# else if env.orientation == 0, horizontal search
			else:
				obs, reward, done, trunc, info, num_iterations, steps = self.horizontalSearch(env, sorted, tot_steps, tot_num_iterations, tot_rewards)
				# record shape's search outputs as a running total
				tot_obs.append(obs)
				tot_info.append(info)
				tot_rewards.append(reward)
				tot_num_iterations += num_iterations
				tot_steps += steps
			
			# save the shape's current location
			end = env.location.copy()

			# if a search returns a done, close env
			if done:
				env.close()

		# if all shapes have been searched, close env
		env.close()

	# computes the steps needed to move from a starting coordinate to and ending coordinate using
	# the environment's step function to mimic a real drone moving between search areas incurring a cost
	def moveToNewShape(self, env, goal):
		tot_obs = []
		tot_info = []
		tot_rewards = 0
		tot_num_iterations = 0
		tot_steps = 0
		trunc = None
		
		dx = None
		dy = None

		env.cfg.WORLD_XS = (env.cfg.WORLD_XS[0], goal[0])
		env.cfg.WORLD_YS = (env.cfg.WORLD_YS[0], goal[1])

		# Compute the differences in x and y coordinates
		dx = goal[0] - env.location[0] if goal[0] > env.location[0] else env.location[0] - goal[0]
		dy = goal[1] - env.location[1] if goal[1] > env.location[1] else env.location[1] - goal[1]

        # Move the drone to the new shape's start location
		while abs(env.location[0] - goal[0]) > 0 or abs(env.location[1] - goal[1]) > 0:
            # Determine the movement in the x and y directions
			action_x = max(min(dx, -env.visible_x * (1 - env.cfg.OVERLAP)), min(dx, env.visible_x * (1 - env.cfg.OVERLAP)))
			action_y = max(min(dy, env.visible_y * (1 - env.cfg.OVERLAP)), min(dy, -env.visible_y * (1 - env.cfg.OVERLAP)))
			action = (action_x, action_y, 0)

            # Perform the movement step
			obs, reward, done, trunc, info = env.step(action)
			tot_obs.append(obs)
			tot_info.append(info)
			tot_rewards += reward
			tot_num_iterations += 1
			tot_steps += 1

            # Update delta_x and delta_y based on the remaining distance
			dx = goal[0] - env.location[0] if goal[0] > env.location[0] else env.location[0] - goal[0]
			dy = goal[1] - env.location[1] if goal[1] > env.location[1] else env.location[1] - goal[1]

			# if step returns done as true, break
			if done:
				break

            # Break if the drone cannot move further
			if action_x == 0 and action_y == 0:
				break

            # Break if the movement has reached the destination
			if goal[0] == env.location[0] and goal[1] == env.location[1]:
				break

		# return the totals from moving from start to end
		return tot_obs, tot_rewards, trunc, tot_info, tot_num_iterations, tot_steps

	def horizontalSearch(self, env, lines, steps, num_iterations, rewards):
		trunc = False
		info = {}
		reward = 0
		obs = None
		done = None
		
		# left-to-right: LTR: (->) & -LTR: (<-)
		LTR = 1

		for i in range(num_iterations):

			# bound[0] = lower bound (towards origin) and bound[1] = upper bound (away from origin)
			bound = []

			env.cfg.WORLD_XS = (env.area_bounds["minx"], env.area_bounds["maxx"])
			env.cfg.WORLD_YS = (env.area_bounds["miny"], env.area_bounds["maxy"])

			obs, reward, done, trunc, info = env.step([5, 5, 0])

			while(abs(env.location[0] - env.end[0]) > 5 or abs(env.location[1] - env.end[1]) > 5) and not env.done:
				if LTR == 1:
					# get the bound for shape in +x direction
					# y is static, use it to solve for intersection to find x border
					bound = self.getBounds(lines[1], env.location[1], 1)

					# while current location is within bounds
					while env.location[0] < max(bound) and not env.done:
						env.cfg.WORLD_XS = (env.cfg.PADDING_X + min(bound), max(bound))
						obs, reward, done, trunc, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0 ,0])
						steps += 1
						rewards.append(reward)

				if LTR == -1:
					# get the bounds for shape in -x direction
					bound = self.getBounds(lines[1], env.location[1], 1)

					while env.location[0] > min(bound) and not env.done:
						env.cfg.WORLD_XS = (min(bound), env.cfg.WORLD_XS[1])
						obs, reward, done, trunc, info = env.step([-env.visible_x  * (1 - env.cfg.OVERLAP), 0 ,0 ])
						steps += 1
						rewards.append(reward)

				LTR = -LTR
				
				# get the bounds for y movement (x is static) as we are now
				# moving vertical instead of horizontal
				if LTR:
					bound = self.getBounds(lines[0], min(env.location[0], bound[1]), 0)
				else:
					bound = self.getBounds(lines[0], max(env.location[0], bound[1]), 0)

				if not env.done:
					env.cfg.WORLD_YS = (env.cfg.WORLD_YS[0] + env.area_bounds["miny"], env.area_bounds["maxy"])
					obs, reward, done, trunc, info = env.step([0, env.visible_y  * (1 - env.cfg.OVERLAP), 0])
				else:
					break

			if env.done:
				break
			else:
				num_iterations += 1

		return obs, reward, done, trunc, info, num_iterations, steps
	
	def verticalSearch(self, env, lines, steps, num_iterations, rewards):
		trunc = False
		info = {}
		reward = 0
		obs = None
		done = None
		
		# Up-to-Down: UTD: (^) & -UTD: (⌄)
		UTD = 1

		for i in range(num_iterations):

			# bound[0] = lower bound (towards origin) and bound[1] = upper bound (away from origin)
			bound = []

			env.cfg.WORLD_XS = (env.area_bounds["minx"], env.area_bounds["maxx"])
			env.cfg.WORLD_YS = (env.area_bounds["miny"], env.area_bounds["maxy"])

			obs, reward, done, trunc, info = env.step([5, 5, 0])

			# Vertical Movement
			while(abs(env.location[0] - env.end[0]) > 5 or abs(env.location[1] - env.end[1]) > 5) and not env.done:
				if UTD == 1:

					bound = self.getBounds(lines[0], env.location[0], 0)

					while env.location[1] < max(bound) and not env.done:
						env.cfg.WORLD_YS = (env.cfg.PADDING_Y, max(bound))
						obs, reward, done, trunc, info = env.step([0, env.visible_y * (1 - env.cfg.OVERLAP), 0])
						steps += 1
						rewards.append(reward)

				if UTD == -1:
					bound = self.getBounds(lines[0], env.location[0], 0)

					while env.location[1] > min(bound) and not env.done:
						env.cfg.WORLD_YS = (min(bound), env.cfg.WORLD_YS[1])
						obs, reward, done, trunc, info = env.step([0, -env.visible_y * (1 - env.cfg.OVERLAP), 0])
						steps += 1
						rewards.append(reward)

				UTD = -UTD

				# Horizontal Movement
				# get the bounds for y movement ("env.orientation ^ 1" will return  opposite orientation) as we are now
				# moving vertical instead of horizontal
				bound = self.getBounds(lines[1], env.location[1], 1)

				if not env.done:
					env.cfg.WORLD_XS = (env.cfg.WORLD_XS[0], env.area_bounds["maxx"])
					obs, reward, done, trunc, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0, 0])

			if env.done:
				break
			else:
				num_iterations += 1

		return obs, reward, done, trunc, info, num_iterations, steps

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run complete coverage against a search area.')
	parser.add_argument('-f', '--filename', type=str, help='Specify the filename: Non-.npy files will be processed before running')
	parser.add_argument('-r', '--random', action='store_true', help='Randomly generate a world to search.')
	parser.add_argument('-s', '--setSize', nargs=2, help='Specify a size (used with random).')
	parser.add_argument('-p', '--process', action='store_true', help='Process a file: Requires either a -f filename or -r randomly generated world')
	parser.add_argument('-tp', '--togglePath', action='store_true', help='Show the path of the drone and output it to a file')
	parser.add_argument('-tb', '--toggleBounds', action='store_true', help='Show the search area\'s boundaries')
	parser.add_argument('-tbat', '--toggleBattery', action='store_true', help='Turn off battery for testing.')
	parser.add_argument('-mw', '--maxWind', type=float, nargs=2, help='Set max wind speed (x, y) in m/s')
	parser.add_argument('-t', '--test', action='store_true', help='sets full toggle on all options for testing')

	args = parser.parse_args()

	Complete_Coverage(args)