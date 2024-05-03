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
import sys
import os

class WorldGenerator():
    def __init__(self, file_name=None, world_y=None, world_x=None):
        self.cfg = Configs()

    def run(self, file_name=None, world_y=None, world_x=None):
        if world_y is not None and world_x is not None and isinstance(world_y, (int, float)) and isinstance(world_x, (int, float)):
            self.cfg.WORLD_YS = (self.cfg.WORLD_YS[0], world_y)
            self.cfg.WORLD_XS = (self.cfg.WORLD_XS[0], world_x)

        if file_name is not None:
            self.name = file_name.rsplit(".", 1)
        else:
            self.name = 'random_world'


        # generate a new world
        WorldGenerator.gen_world(self)

        # save points of "shape" as numpy array
        np.save(self.name + "_points", self.points)
        # save as numpy array and png
        np.save(self.name, self.world)
        gray_scaled = (self.world * 255).astype(np.uint8)
        cv2.imwrite(self.name + ".png", gray_scaled)
        print("\nWorld saved as " + self.name + "\n")

        # return the name of the image created
        return self.name + ".png"

    @staticmethod
    def gen_world(self):
        """"
        Generates a new world for the drone based on size and # seeds specified in configurations.py
        """

        # The padded area of the world is were the drone cannot go to but may appear in the frame
        seeds = self.cfg.SEEDS

        # tuple representing dimensions of world
        y = self.cfg.WORLD_YS[1]
        x = self.cfg.WORLD_XS[1]

        size = (y + self.cfg.PADDING_Y, x + self.cfg.PADDING_X)

        self.points = [[[self.cfg.PADDING_X, self.cfg.PADDING_Y], [self.cfg.PADDING_X, y], [x, y], [x, self.cfg.PADDING_Y], [self.cfg.PADDING_X, self.cfg.PADDING_Y]]]

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

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        print("Usage: python3 world_generator.py [file_name] [world_y] [world_x]")
        sys.exit(0)

    file_name = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        world_y = int(sys.argv[2]) if len(sys.argv) > 2 else None
        world_x = int(sys.argv[3]) if len(sys.argv) > 3 else None

    except ValueError:
        print("Error: world_y and world_x must be integers.")
        print("Usage: python3 world_generator.py [file_name] [world_y] [world_x]")

        sys.exit(1)

    wg_instance = WorldGenerator().run(file_name, world_y, world_x)
        