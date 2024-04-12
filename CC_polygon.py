from drone_environment import droneEnv
import numpy as np
import argparse
import math
from world_generator import WorldGenerator
from process_image import process

#TODO: 
	# 1. check getBounds to make sure it returns the correct value
	# 2. Connect shapes in some manner that makes sense
		# * get end point of current shape, find start point of next shape
		# * then have it follow a direct line from shape 1 to shape 2
	# 3. shapes might be defined in an unorderd manner, we need to reorder them
	# 4. so that the travel between shapes is direct and does not pass over others

class Complete_Coverage():
	def __init__(self, args):

		wname = None

		x = 0
		y = 0

		# if given a filename argument
		if args.filename:
			wname = args.filename
			# if that filename is not a npy array, process image w/ factor of 1
			# TODO: Maybe make a flag that can set factor for process?
			# or just set factor to be anything over x sized (anything over 10000x10000 gets factored?)
			if '.npy' not in args.filename:
				process(args.filename, 1)
				wname = args.filename.rsplit(".", 1)[0] + ".npy"
		# if given a filename to name a random world
		if args.randomSetFilename:
			# generate that file
			WorldGenerator().run(file_name=args.randomSetFilename)
			wname = args.randomSetFilename
		# if given no name and request for random world
		if args.random:
			# generate a random world with name 'random_world'
			WorldGenerator().run(file_name='random_world')
			wname = 'random_world.npy'
		# if given no filename but request certain size of random world
		if args.randomSetSize:
			# generate random world of x, y size named 'random_world'
			x, y = args.randomSetSize
			WorldGenerator().run(file_name='random_world', world_y=int(y) , world_x=int(x))
			wname = 'random_world.npy'
		# if set size and filename is given, set size of world and name
		if args.randomSetSizeFilename:
			wname, x, y = args.randomSetSizeFilename
			WorldGenerator().run(file_name=wname, world_y=int(y), world_x=int(x))

		# init environment
		env = droneEnv(render=True, generate_world=False, wname=wname, learn=False)

		# if we are setting a specific size of map, override cfg's world sizes
		if x != 0 and y != 0:
			env.cfg.WORLD_XS = (env.cfg.PADDING_X, int(x))
			env.cfg.WORLD_YS = (env.cfg.PADDING_Y, int(y))

		# load the world from file
		env.world = np.load(env.world_name)
		
		# remove all padding and optimize the order of shapes
		self.sanitize(env)

		# run search
		self.search(env)

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

	def findStart(Self, list, cord):
		return np.min(list[0][:, cord])

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
				while left >= curr and self.checkValue(lines[left], val, cord):
					first_line = left
					left -= 1

				# Search for the last occurrence on the right
				right = mid + 1
				while right <= end and self.checkValue(lines[right], val, cord):
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
		# get upper and lower bound lines in respect to constant x or y given
		upper, lower = self.binarySearch(lines, val, cord)

		# if there aren't two lines return None (Should never happen)
		if upper is None or lower is None:
			# TODO: maybe raise an exception instead of returning None as this propogates into 
			# an unrecoverable error??
			return None

		# find the upper/lower of both x and y for both end points in both lines
		upper_x1, upper_y1 = upper[0]
		upper_x2, upper_y2 = upper[1]
		# lower being the lower bound line
		lower_x1, lower_y1 = lower[0]
		lower_x2, lower_y2 = lower[1]

		# check if the lines are vertical or horizontal
		if upper_x1 == upper_x2 and lower_x1 == lower_x2:  # Vertical lines
            # handle the case where both lines are vertical
			if cord == 0:
				return upper_x1, lower_x1
			else:
				return max(upper_y1, lower_y1), min(upper_y2, lower_y2)
		elif upper_y1 == upper_y2 and lower_y1 == lower_y2:  # Horizontal lines
            # handle the case where both lines are horizontal
			if cord == 0:
				return min(upper_x1, lower_x1), max(upper_x2, lower_x2)
			else:
				return upper_y1, lower_y1

		# calc the slope (m) and intercept (b) for each line
		upper_m = (upper_y2-upper_y1)/float((upper_x2-upper_x1))
		upper_b = upper_y1 - upper_m * upper_x1

		lower_m = (lower_y2-lower_y1)/float((lower_x2-lower_x1))
		lower_b = lower_y1 - lower_m * lower_x1

		# solve for x value
		if cord == 0:
			upper_bound = (val - upper_b) / upper_m
			lower_bound = (val - lower_b) / lower_m

			return math.ceil(upper_bound), math.ceil(lower_bound)
		# solve for y value
		else:
			upper_bound = upper_m * val + upper_b
			lower_bound = lower_m * val + lower_b

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
			area_bounds = self.findMinMax(shape)

			# get lines sorted by x or y ranges
			# sorted[0] = sorted by x range, sorted[1] = sorted by y range
			sorted = self.getLines(shape)

			# 0 means LTR, 1 means UTD
			env.orientation = self.getOrientation(area_bounds)

			# sets start x to the first index of the sorted xsorted line list's x value
			env.location[0] = max(self.findStart(sorted[env.orientation], 0), env.cfg.PADDING_X)
			# sets start y to the first index of the sorted xsorted line list's y value
			env.location[1] = max(self.findStart(sorted[env.orientation], 1), env.cfg.PADDING_Y)

			print(env.location)

			# if we have seached at least one shape, move to our next shape via step
			# to record costs of moving between search areas
			if end is not None:
				self.moveToNewShape(env, end, env.location)

			# if env.orientation == 1, vertical search
			if env.orientation:
				obs, reward, done, trunc, info, num_iterations, steps = self.verticalSearch(env, sorted[env.orientation], tot_steps, tot_num_iterations, tot_rewards)
				# record shape's search outputs as a running total
				tot_obs.append(obs)
				tot_info.append(info)
				tot_rewards.append(reward)
				tot_num_iterations += num_iterations
				tot_steps += steps
			# else if env.orientation == 0, horizontal search
			else:
				obs, reward, done, trunc, info, num_iterations, steps = self.horizontalSearch(env, sorted[env.orientation], tot_steps, tot_num_iterations, tot_rewards)
				# record shape's search outputs as a running total
				tot_obs.append(obs)
				tot_info.append(info)
				tot_rewards.append(reward)
				tot_num_iterations += num_iterations
				tot_steps += steps
			
			# save the shape's current location
			end = env.location

			# if a search returns a done, close env
			if done:
				env.close()

		# if all shapes have been searched, close env
		env.close()

	# computes the steps needed to move from a starting coordinate to and ending coordinate using
	# the environment's step function to mimic a real drone moving between search areas incurring a cost
	def moveToNewShape(self, env, start, end):
		tot_obs = []
		tot_info = []
		tot_rewards = 0
		tot_num_iterations = 0
		tot_steps = 0

        # Compute the differences in x and y coordinates
		delta_x = end[0] - start[0]
		delta_y = end[1] - start[1]

        # Move the drone to the new shape's start location
		while abs(env.location[0] - end[0]) > 0 or abs(env.location[1] - end[1]) > 0:
            # Determine the movement in the x and y directions
			action_x = max(min(delta_x, -self.visible_x * (1 - self.cfg.OVERLAP)), self.visible_x * (1 - self.cfg.OVERLAP))
			action_y = max(min(delta_y, self.visible_y * (1 - self.cfg.OVERLAP)), -self.visible_y * (1 - self.cfg.OVERLAP))
			action = (action_x, action_y, 0)

            # Perform the movement step
			obs, reward, done, trunc, info = self.step(action)
			tot_obs.append(obs)
			tot_info.append(info)
			tot_rewards += reward
			tot_num_iterations += 1
			tot_steps += 1

            # Update delta_x and delta_y based on the remaining distance
			delta_x = end[0] - env.location[0]
			delta_y = end[1] - env.location[1]

			# if step returns done as true, break
			if done:
				break

            # Break if the drone cannot move further
			if action_x == 0 and action_y == 0:
				break

            # Break if the movement has reached the destination
			if env.location[0] == end[0] and env.location[1] == end[1]:
				break

		# return the totals from moving from start to end
		return tot_obs, tot_rewards, trunc, tot_info, tot_num_iterations, tot_steps

	def horizontalSearch(self, env, lines, steps, num_iterations, rewards):
		trunc = False
		info = {}
		
		# left-to-right: LTR: (->) & -LTR: (<-)
		LTR = 1

		for i in range(num_iterations):

			# bound[0] = lower bound (towards origin) and bound[1] = upper bound (away from origin)
			bound = []

			while True and not env.done:
				if LTR == 1:
					# get the bound for shape in +x direction
					# y is static, use it to solve for intersection to find x border
					bound = self.getBounds(lines, env.location[1], env.orientation)

					# while current location is within bounds
					while abs(env.location[0] - (bound[1] - env.cfg.PADDING_X)) > 1 and not env.done:
						obs, reward, done, trunc, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0 ,0])
						steps += 1
						rewards.append(reward)

				if LTR == -1:
					# get the bounds for shape in -x direction
					bound = self.getBounds(lines, env.location[0], env.orientation)

					while abs(env.location[0] - (bound[0] + env.cfg.PADDING_X)) > 1 and not env.done:
						obs, reward, done, trunc, info = env.step([-env.visible_x  * (1 - env.cfg.OVERLAP), 0 ,0 ])
						steps += 1
						rewards.append(reward)

				LTR = -LTR
				
				# get the bounds for y movement ("env.orientation ^ 1" will return  opposite orientation) as we are now
				# moving vertical instead of horizontal
				bound = self.getBounds(lines, env.location[1], env.orientation ^ 1)

				if abs(env.location[1] - bound[1]) > 1 and not env.done:
					obs, reward, done, trunc, info = env.step([0, env.visible_y  * (1 - env.cfg.OVERLAP), 0])
				else:
					break

			num_iterations += 1

		return obs, reward, done, trunc, info, num_iterations, steps
	
	def verticalSearch(self, env, lines, steps, num_iterations, rewards):
		trunc = False
		info = {}
		
		# Up-to-Down: UTD: (^) & -UTD: (⌄)
		UTD = 1

		for i in range(num_iterations):

			# bound[0] = lower bound (towards origin) and bound[1] = upper bound (away from origin)
			bound = []

			# Vertical Movement
			while True and not env.done:
				if UTD == 1:

					bound = self.getBounds(lines, env.location[1], env.orientation)

					while abs(env.location[1] - (bound[0] - env.cfg.PADDING_Y)) > 1 and not env.done:
						obs, reward, done, trunc, info = env.step([0, env.visible_y * (1 - env.cfg.OVERLAP), 0])
						steps += 1
						rewards.append(reward)

				if UTD == -1:
					bound = self.getBounds(lines, env.location[1], env.orientation)

					while abs(env.location[1] - (bound[1] + env.cfg.PADDING_Y)) > 1 and not env.done:
						obs, reward, done, trunc, info = env.step([0, -env.visible_y * (1 - env.cfg.OVERLAP), 0])
						steps += 1
						rewards.append(reward)

				UTD = -UTD

				# Horizontal Movement
				# get the bounds for y movement ("env.orientation ^ 1" will return  opposite orientation) as we are now
				# moving vertical instead of horizontal
				bound = self.getBounds(lines, env.location[0], env.orientation)

				if abs(env.location[0] - bound[1]) > 1 and not env.done:
					obs, reward, done, trunc, info = env.step([env.visible_x * (1 - env.cfg.OVERLAP), 0, 0])

			num_iterations += 1

		return obs, reward, done, trunc, info, num_iterations, steps

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