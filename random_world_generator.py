# -*- coding: utf-8 -*-
"""
Created on Wed January 12 03:18:32 2024

@author: cmdraper

Abstracted world generator for env based off the original method in drone_environment 
written by aniaraki and modified to stand alone by cmdraper
"""

import numpy as np
import cv2
import random
from configurations import Configs

class world_generator():
    def __init__(self):
        self.cfg = Configs()

    def gen_world(self):
        """"
        Generates a new world for the drone based on size and # seeds specified in configurations.py
        """

        # The padded area of the world is were the drone cannot go to but may appear in the frame
        seeds = self.cfg.SEEDS

        # tuple representing dimensions of world
        size = (self.cfg.WORLD_YS[1] + self.cfg.PADDING_Y, self.cfg.WORLD_XS[1] + self.cfg.PADDING_X)

        # initialize world with zeros
        self.world = np.zeros(size, dtype=int)

        square_corners = []
        for s in range(seeds):
            # Corner of each square corner=[x,y] (PLACES REWARDS IN PADDING)
            # corner=[random.randint(-self.cfg.PADDING_X, self.cfg.WORLD_XS[1]) + self.cfg.PADDING_X,
            #         random.randint(-self.cfg.PADDING_Y, self.cfg.WORLD_YS[1]) + self.cfg.PADDING_Y]
             
            corner = [random.randint(0, self.cfg.WORLD_XS[1] - self.cfg.PADDING_X) + self.cfg.PADDING_X,
                      random.randint(0, self.cfg.WORLD_YS[1] - self.cfg.PADDING_Y) + self.cfg.PADDING_Y]

             # List of all square corners
            square_corners.append(corner)
            square_size = random.randint(self.cfg.square_size_range[0], self.cfg.square_size_range[1])
            for i in range(square_size):
                for j in range(square_size):
                    try:
                        self.world[corner[1] + j][corner[0] + i] = 1
                    except:
                        pass

        return self.save_world()

    def save_world(self):
        # save as numpy array and png
        name = input('Please name this new world: ')
        np.save(name, self.world)
        cv2.imwrite(name + ".png", self.world * 255)
        print("\nWorld saved as " + name + "\n")

        return name + ".npy"

if __name__ == "__main__":
    wg = world_generator()
        