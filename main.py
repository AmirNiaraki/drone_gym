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
    parser.add_argument('--detector', type=str, choices=['retina', 'low_fidelity', 'double_clustering'],
                        default='keyboard', help='Type of inference model to use for object localization')
    parser.add_argument('--show_location', type=bool, default=False, help='Print out the location of the drone')
    return parser.parse_args()

def main(image_path, navigator_type, show_location=False, model_type='retina'):
    logging.info(f"Using image: {image_path}")
    logging.info(f"Using navigator: {navigator_type}")
    env = initialize_env(image_path, show_location)
    drone_data = {
        "step_count": [],
        "battery_levels": [],
        "anomaly_areas": [],
        "percentages_of_anomaly": [],
        "locations": [],
        "world_x1": [],
        "world_y1": [],
        "world_x2": [],
        "world_y2": [],
    }
    # for navigation
    if navigator_type == 'complete':
        navigator = CompleteCoverageNavigator(env)
    else:
        navigator = KeyboardNavigator(env)
    # for inference
    model = Inferer(env.cfg, model_type)
    for obs, info in navigator.navigate():
        # Process the observation
        logging.info(f"Observation: {obs.shape}")
        logging.info(f"info: {info}")

        log_location(model, obs, info, drone_data)
        # boxes = model.infer(obs)
        # env.update_boxes(boxes) # allows drawing of boxes on the image

        # if boxes is not []:
        #     logging.info(f"Detected {boxes}")
        # env.update_boxes(boxes) # allows drawing of boxes on the image

        # cv2.imwrite('images/observation.jpg', obs)

def log_location(model, obs, info, drone_data):
    # get the needed information from env
    location = info["location"]
    boundaries = info["boundaries"]
    scale_x = info["scale_x"]
    scale_y = info["scale_y"]
    boxes = model.infer(obs)
    step_count = info["step_count"]
    battery = info["battery_levels"]

    # TODO: need to check if there is any conversion between the
    # inference frame and the big frame
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])

        # Convert to world coordinates
        world_x1 = int(x1 * scale_x + boundaries[2])
        world_y1 = int(y1 * scale_y + boundaries[0])
        world_x2 = int(x2 * scale_x + boundaries[2])
        world_y2 = int(y2 * scale_y + boundaries[0])

        logging.info(f"Locations {world_x1, world_y1, world_x2, world_y2}")
        logging.info(f"Original locations {x1, y1, x2, y2}")

        # Store values in the evaluation dictionary
        drone_data["step_count"].append(step_count)
        drone_data["battery_levels"].append(battery)
        drone_data["locations"].append(location)

        # Save world coordinates
        drone_data["world_x1"].append(world_x1)
        drone_data["world_y1"].append(world_y1)
        drone_data["world_x2"].append(world_x2)
        drone_data["world_y2"].append(world_y2)

def initialize_env(input_map, show_location):
    env = droneEnv(observation_mode='cont', action_mode='cont', render=True, img_path=input_map)
    env.cfg.show_location =show_location
    return env

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args.image_path, args.navigator, args.show_location, args.detector)
