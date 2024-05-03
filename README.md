## Requirements
To install requirements, run: `pip install requirements.txt`

## Complete Coverage
### Overview
Simulates a drone flying a naive pass of the world 

### To Run Without Flags (Run on a Generated World)
`python3 complete_coverage.py`

### To Run With Flags

`Python3 CC_polygon.py [-h] [-f FILENAME] [-r] [-s SETSIZE SETSIZE] [-p] [-tp] [tb] [-tbat] [-mw x y] [-t]`

options:
  `-h, --help`           
  		show this help message and exit
  `-f FILENAME, --filename FILENAME`
		Use a filename (if file extension is not ".npy", a random world will be generated and saved under the given name)
  `-r, --random`          
  		Randomly generate a world to search.
  `-s X Y, --setSize X Y`
		Specify a size (used with random).
  `-p, --process`         
  		Process a file or randomly generated world
  `-tp, --togglePath`     
  		Show the path of the drone and output it to a png file
  `-tb, --toggleBounds`   
  		Show the search area's boundaries
  `-tbat, --toggleBattery`
		Turn off battery for testing.
  `-mw X Y, --maxWind X Y`
		Set max wind speed (x, y) in m/s
  `-t, --test`           
		sets full toggle on all options for testing`python3 process_image.py <.png file>

Example:
	To randomly generate a world of size 600 x 400, process it, and show path and bounds without battery
		`python3 CC_polygon.py -r -p -t -s 600 400` 
	To load a saved world and process it, setting bounds and path
		`python3 CC_polygon.py -f <filename>.npy -p -tp -tb

## Reinforced Learning

### To Run

`tensorboard --logdir=Training/Log`

# Simulation Notes
## On the drag coefficient and flight duration (Amir)
The drag coefficient is setup such that the drone flies for an average of 4 km, with the speed of 10 m/s, in an average wind of 3.5 fm/s before it runs out of battery.
