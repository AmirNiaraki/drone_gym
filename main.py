"""
This script initializes a droneEnv and receives the observation (drone view)
At every step regardless of the navigator and gives the observation to the inference model.

Usage:
    python main.py --image_path <path_to_image> --navigator <navigator_type>

Example:
    python main.py --image_path images/sample2.png --navigator complete
    python main.py --image_path images/sample2.png --navigator keyboard
"""

import argparse
import json
import logging
import time

import cv2
import numpy as np

from drone_environment import droneEnv
from inference import Inferer
from navigator import (
    CompleteCoverageNavigator,
    HierarchicalNavigator,
    KeyboardNavigator,
)

# from CCdrone import CCdrone


def parse_args():
    parser = argparse.ArgumentParser(description="Drone navigation script")
    parser.add_argument(
        "--image_path",
        type=str,
        default="images/sample.png",
        help="Path to the input image",
    )
    parser.add_argument(
        "--navigator",
        type=str,
        choices=["complete", "keyboard", "hierarchical"],
        default="keyboard",
        help="Type of navigator to use",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["retina", "low_fidelity", "double_clustering"],
        default="low_fidelity",
        help="Type of inference model to use for object localization",
    )
    parser.add_argument(
        "--show_location",
        type=bool,
        default=False,
        help="Print out the location of the drone",
    )
    parser.add_argument(
        "--is_post_process",
        help="Flag to determine if recordings are going to be unencoded.",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


def main(image_path, navigator_type, show_location=False, model_type="retina", is_post_process=False):
    if not is_post_process:
        logging.info(f"Using image: {image_path}")
        logging.info(f"Using navigator: {navigator_type}")
        env = initialize_env(image_path, show_location)
        drone_info_dict = {
            "image_path": image_path,
            "step_count": [],
            "battery_levels": [],
            "locations": [],
            "world_x1": [],
            "world_y1": [],
            "world_x2": [],
            "world_y2": [],
        }
        # for navigation
        if navigator_type == "complete":
            navigator = CompleteCoverageNavigator(env)
        elif navigator_type == "keyboard":
            navigator = KeyboardNavigator(env)
        elif navigator_type == "hierarchical":
            navigator = HierarchicalNavigator(env)
        # for inference
        model = Inferer(env.cfg, model_type)
        for obs, info in navigator.navigate():
            # Process the observation
            score, boxes = model.infer(obs)
            env.info_list[-1]['detection_score'] = score

            log_location(model, boxes, obs, info, drone_info_dict)
    else:
        # Open and read the JSON file
        with open("data_info.json", "r") as file:
            drone_info_dict = json.load(file)
        post_process(drone_info_dict)


def iou(box1, box2):
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union
    union = area1 + area2 - intersection

    # Return IoU
    return intersection / union if union > 0 else 0


def merge_overlapping_boxes(boxes, iou_threshold=0.5):
    merged_boxes = []
    while boxes:
        # Take the first box
        current_box = boxes.pop(0)
        x1, y1, x2, y2 = current_box[:4]

        # Group all overlapping boxes
        overlapping_boxes = [current_box]
        remaining_boxes = []
        for box in boxes:
            if iou(current_box[:4], box[:4]) >= iou_threshold:
                overlapping_boxes.append(box)
            else:
                remaining_boxes.append(box)

        # Merge overlapping boxes into one
        x1 = min(box[0] for box in overlapping_boxes)
        y1 = min(box[1] for box in overlapping_boxes)
        x2 = max(box[2] for box in overlapping_boxes)
        y2 = max(box[3] for box in overlapping_boxes)

        # Append the merged box
        merged_boxes.append([x1, y1, x2, y2])
        boxes = remaining_boxes

    return merged_boxes


def post_process(drone_info):
    step_count = drone_info["step_count"]
    world_x1 = drone_info["world_x1"]
    world_y1 = drone_info["world_y1"]
    world_x2 = drone_info["world_x2"]
    world_y2 = drone_info["world_y2"]
    image_path = drone_info["image_path"]

    # Combine bounding boxes with step counts into a single list
    boxes = [
        [world_x1[i], world_y1[i], world_x2[i], world_y2[i], step_count[i]]
        for i in range(len(step_count))
    ]

    # Merge overlapping boxes
    merged_boxes = merge_overlapping_boxes(boxes, iou_threshold=0.2)

    # Open the image
    img = cv2.imread(image_path)

    # Draw the merged bounding boxes
    for box in merged_boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Merged boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Save the output image
    cv2.imwrite("output.png", img)
    logging.info(f"Output image saved at: output.png")


def log_location(model, boxes, obs, info, drone_info_dict):
    # get the needed information from env
    location = info["location"]
    boundaries = info["boundaries"]
    scale_x = info["scale_x"]
    scale_y = info["scale_y"]
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
        drone_info_dict["step_count"].append(step_count)
        drone_info_dict["battery_levels"].append(battery)
        drone_info_dict["locations"].append(location)

        # Save world coordinates detections
        drone_info_dict["world_x1"].append(world_x1)
        drone_info_dict["world_y1"].append(world_y1)
        drone_info_dict["world_x2"].append(world_x2)
        drone_info_dict["world_y2"].append(world_y2)

    # so that we do not write to file if there are no detections
    if len(boxes) == 0:
        with open("data_info.json", "w") as file:
            json.dump(drone_info_dict, file, indent=4)


def initialize_env(input_map, show_location):
    env = droneEnv(
        observation_mode="cont", action_mode="cont", render=True, img_path=input_map
    )
    env.cfg.show_location = show_location
    return env


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args.image_path, args.navigator, args.show_location, args.detector, args.is_post_process)
