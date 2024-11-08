'''
This script is the inference engine containing multiple objects (each for one model)
that receive a frame and return bounding box of the detected objects within the frames.
There are three models:
1- low_fidelity: it expects a gray scale image and counts the black pixels in the frame
2- DCNN: is a double clustering method that bins the pixels of a certain color and then
         clusters the bins of the same class that are next to each other and returns bb
3- retinaNet: is a deep learning model that returns bounding boxes of the detected objects
'''

import cv2
import logging

class Inferer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = 'low_fidelity'
        self.remove_battery_bar = True
        logging.info(f'Using model: {self.model}')

    def infer(self, frame):
        # TODO: remove the battery bar from frame entirely in the project
        if self.remove_battery_bar:

            frame=frame[0:self.cfg.FRAME_H,0:self.cfg.FRAME_W]
            # self.write_sample_to_disk(frame)

    def write_sample_to_disk(self, frame):
        cv2.imwrite('images/frame_for_inference.jpg', frame)


