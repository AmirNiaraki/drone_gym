'''
This script initializes a droneEnv and receives the observation (drone view)
At every step regardless of the navigator and gives the observation to the inference model.

Usage:
    python main.py --image_path <path_to_image> --navigator <navigator_type>

Example:
    python main.py --image_path images/sample2.png --navigator complete
    python main.py --image_path images/sample2.png --navigator keyboard
'''

from drone_environment import droneEnv
from navigator import CompleteCoverageNavigator, KeyboardNavigator
import argparse
import logging
import time
import cv2
import numpy as np
from inference import Inferer
# from CCdrone import CCdrone

def parse_args():
    parser = argparse.ArgumentParser(description="Drone navigation script")
    parser.add_argument('--image_path', type=str, default='images/sample.png', help='Path to the input image')
    parser.add_argument('--navigator', type=str, choices=['complete', 'keyboard'], default='keyboard', help='Type of navigator to use')
    return parser.parse_args()

def main(image_path, navigator_type):
    logging.info(f"Using image: {image_path}")
    logging.info(f"Using navigator: {navigator_type}")
    env = initialize_env(image_path)
    # for navigation
    if navigator_type == 'complete':
        navigator = CompleteCoverageNavigator(env)
    else:
        navigator = KeyboardNavigator(env)
    # for inference
    model = Inferer(env.cfg)
    for obs in navigator.navigate():
        # Process the observation
        logging.info(f"Observation: {obs.shape}")
        model.infer(obs)

        # cv2.imwrite('images/observation.jpg', obs)

def initialize_env(input_map):
    env = droneEnv(observation_mode='cont', action_mode='cont', render=True, img_path=input_map)
    return env

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args.image_path, args.navigator)