"""
This script is the inference engine containing multiple objects (each for one model)
that receive a frame and return bounding box of the detected objects within the frames.
There are three models:
1- low_fidelity: it expects a gray scale image and counts the black pixels in the frame
2- DCNN: is a double clustering method that bins the pixels of a certain color and then
         clusters the bins of the same class that are next to each other and returns bb
3- retinaNet: is a deep learning model that returns bounding boxes of the detected objects
"""

import logging

import cv2
import numpy as np
import torch


class Inferer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = "low_fidelity"

        logging.info(f"Using model: {self.model}")

    def infer(self, frame):
        # TODO: remove the battery bar from frame entirely in the project
        # count the number of black pixels within the frame
        if self.model == "low_fidelity":
            cv2.imwrite("images/frame_for_inference.jpg", frame)
            score = self.count_black_pixels(frame)

    def count_black_pixels(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_pixels = gray.size - cv2.countNonZero(gray)
        logging.info(f"Number of black pixels: {black_pixels}")
        return black_pixels

    def write_sample_to_disk(self, frame):
        cv2.imwrite("images/frame_for_inference.jpg", frame)
