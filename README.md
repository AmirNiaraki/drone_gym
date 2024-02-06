## Requirements
To install requirements, run: `pip install requirements.txt`

## Complete Coverage
### Overview
Simulates a drone flying a naive pass of the world 

### To Run
`python complete_coverage.py`

`python image_editor.py <.png file>`

`tensorboard --logdir=Training/Log`

# Simulation Notes
## On the drag coefficient and flight duration (Amir)
The drag coefficient is setup such that the drone flies for an average of 4 km, with the speed of 10 m/s, in an average wind of 3.5 fm/s before it runs out of battery.

## To View TensorBoard
`tensorboard --logdir=Training/Logs`

Will need to install tensorboard and tensorflow first.