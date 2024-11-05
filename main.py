'''
This script initializes a droneEnv and receives the observation (drone view)
At every step regardless of the navigator and gives the observation to the inference model.
'''

from drone_environment import droneEnv
import time
import cv2
import numpy as np

