# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:35:20 2022

@author: aniaraki


----> X
|
Y
"""
import logging
import os
import pickle
import random
import sys
import threading
import time
from math import acos, ceil, degrees, radians, sqrt, tan
from sys import exit
from threading import Thread
import json

import cv2
import gymnasium
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

from configurations import Configs

log_level = "INFO"
# Logging is configured to write to droneENV.log AND to stdout
log_format = "[%(asctime)s] [%(name)-4s] [%(filename)20s:%(lineno)-4d] [%(levelname)-8s] - %(message)s"
logging.basicConfig(
    format=log_format,
    level=getattr(logging, log_level),
    filemode="a",
    filename="droneENV.log",
)
root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(getattr(logging, log_level))
handler.setFormatter(logging.Formatter(log_format))
root.addHandler(handler)
###


class droneEnv(gymnasium.Env):

    def __init__(
        self, observation_mode, action_mode, render=False, img_path="images/sample2.png"
    ):
        # super(droneEnv, self).__init__()
        super().__init__()
        self.cfg = Configs(img_path)
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.render = render

        self.image_path = img_path if img_path != None else self.cfg.geotiff_path

        # used to draw the boxes in the big frame
        self.current_boxes = []

        # self.world=None

        self.location = self.cfg.init_location
        self.path = []  # list of all locations visited by the drone
        self.path.append(self.location)
        #TODO: depricate path and infer it from the flight info log
        self.info_list=[]

        logging.info(f"initial location of drone: {self.location}")

        if self.cfg.load_from_geotiff == True:
            self.world = self.load_geotiff()
        else:
            self.world = self.world_genertor()
        if self.cfg.create_explored_map:
            self.explored_map = self.world.copy()
            self.explored_map.fill(0)
            
        self.mask = np.zeros((self.world.shape[0], self.world.shape[1]), dtype=np.uint8)
        self.drone_data = {
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

        # print(type(self.world), self.world.shape)

        ### wind field = (wind_x, wind_y) m/s. with x pointing at east, and positive y pointing at south
        self.wind = (3.5, 0)
        self.battery = self.cfg.FULL_BATTERY  # [x,y,z,] m
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.step_count = 0
        self.battery_inloop = True if self.cfg.battery_inloop else False

        self.drag_normalizer_coef = 0.005

        self.action = [0, 0, 0]
        if self.observation_mode == "cont":
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(self.cfg.FRAME_H, self.cfg.FRAME_W + 1, 1),
                dtype=np.uint8,
            )  # NOTE: dtype has to change for non binary image
            # self.observation_space=Box(low=-2000, high=2000,
            #                            shape=(6,), dtype=np.float64)
        if self.action_mode == "cont":
            self.action_space = Box(
                low=-self.cfg.MAX_SPEED,
                high=self.cfg.MAX_SPEED,
                shape=(3,),
                dtype=np.float16,
            )
        if self.action_mode == "disc":
            self.action_space = Discrete(4)

        if self.observation_mode == "disc":
            ### action list for 2d: [0 ,1       ,2    ,3         ,4   ,5        ,6   ,7]
            ### action list for 2d: [up,up-right,right,right-down,down,down-left,left,left-top ]
            self.action_space = Box(
                low=-self.cfg.MAX_SPEED,
                high=self.cfg.MAX_SPEED,
                shape=(3,),
                dtype=np.float16,
            )
            self.observation_space = Box(
                low=-2000, high=2000, shape=(6,), dtype=np.float16
            )

        ### for getting the frame to the agent at all times
        time.sleep(0.01)
        self.thread = Thread(target=self.update_frame, args=(), daemon=True)
        self.thread.start()
        time.sleep(0.01)
        print("environment is initialized")

    def update_frame(self):
        self.imager_thread_name = threading.current_thread()
        print("top of the thread")
        self.world_img = self.world
        # print("world image shape: ", self.world_img.shape)

        while self.done == False:
            self.visible_x = ceil(tan(radians(self.cfg.FOV_X)) * 2 * self.location[2])
            self.visible_y = ceil(tan(radians(self.cfg.FOV_Y)) * 2 * self.location[2])
            # self.world_img = np.uint8((1 - self.world) * 255) if not self.cfg.load_from_geotiff else self.world
            ### take snap of the sim based on location [x,y,z]
            ### visible corners of FOV in the form boundaries= [y,y+frame_h,x,x+frame_w]
            self.boundaries = [
                int(-self.visible_y / 2 + self.location[1]),
                int(self.visible_y / 2 + self.location[1]),
                int(-self.visible_x / 2 + self.location[0]),
                int(self.visible_x / 2 + self.location[0]),
            ]

            crop = self.world_img[
                self.boundaries[0] : self.boundaries[1],
                self.boundaries[2] : self.boundaries[3],
            ]
            if self.cfg.create_explored_map:
                self.explored_map[
                    self.boundaries[0] : self.boundaries[1],
                    self.boundaries[2] : self.boundaries[3],
                ] = crop
                # cv2.imwrite('images/crop.png',crop)
                # print('wrote images/crop.png for sanity check')
            resized = cv2.resize(crop, (self.cfg.FRAME_W, self.cfg.FRAME_H))

            # # TODO: need to check if there is any conversion between the
            # # inference frame and the big frame
            # for box in self.current_boxes:
            #     logging.info(f"Box: {self.current_boxes}")
            #     # Calculate scaling factors
            #     scale_x = self.visible_x / self.cfg.FRAME_W
            #     scale_y = self.visible_y / self.cfg.FRAME_H
            #
            #     x1, y1, x2, y2 = map(int, box[:4])
            #
            #     # Convert to world coordinates
            #     world_x1 = int(x1 * scale_x + self.boundaries[2])
            #     world_y1 = int(y1 * scale_y + self.boundaries[0])
            #     world_x2 = int(x2 * scale_x + self.boundaries[2])
            #     world_y2 = int(y2 * scale_y + self.boundaries[0])
            #
            #     self.mask[world_y1:world_y2, world_x1:world_x2] = 1
            #     area_of_anomaly = np.sum(self.mask)
            #     percentage_of_anomaly = (
            #         area_of_anomaly / (self.cfg.FRAME_H * self.cfg.FRAME_W)
            #     ) * 100
            #     logging.info(f"Current area of anomaly {area_of_anomaly}")
            #     logging.info(f"Current percentage of anomaly {percentage_of_anomaly}")
            #     logging.info(f"Locations {world_x1, world_y1, world_x2, world_y2}")
            #     logging.info(f"Original locations {x1, y1, x2, y2}")
            #
            #
            #     # Store values in the evaluation dictionary
            #     self.drone_data["step_count"].append(self.step_count)
            #     self.drone_data["battery_levels"].append(self.battery)
            #     self.drone_data["anomaly_areas"].append(area_of_anomaly)
            #     self.drone_data["percentages_of_anomaly"].append(percentage_of_anomaly)
            #     self.drone_data["locations"].append(self.location)
            #
            #     # Save world coordinates
            #     self.drone_data["world_x1"].append(world_x1)
            #     self.drone_data["world_y1"].append(world_y1)
            #     self.drone_data["world_x2"].append(world_x2)
            #     self.drone_data["world_y2"].append(world_y2)
                # print(self.drone_data)

            # cv2.imshow("mask", self.mask * 255)
            self.frame = resized
            self.current_boxes = []

            if self.done == True:
                # print('done is true instide update_frame() trying to join')
                break

        # print('Frame Update stopping...  ',  self.imager_thread_name)

    def update_boxes(self, boxes):
        """Used to update the boxes in the frame"""
        self.current_boxes = boxes

    def fetch_frame(self):
        return self.frame

    def fetch_anomaly(self):
        """
        This method is used to calculate the reward for the agent by counting the black pixels.
        Additionally it sets the observed anomaly to 0 in the world.
        returns:
        score (int); number of black pixels in the frame
        """
        observation = self.fetch_frame()
        # refers to image with no battery
        obs_with_no_battery = observation[0 : self.cfg.FRAME_H, 0 : self.cfg.FRAME_W]
        # scor eis simply the number of black pixels
        score = self.cfg.FRAME_H * self.cfg.FRAME_W - np.sum(
            obs_with_no_battery / 255, dtype=np.int32
        )
        ### removing the detected objects from the world!!!
        if self.cfg.remove_redetected_from_world == True:
            self.world[
                int(-self.visible_y / 2 + self.location[1]) : int(
                    self.visible_y / 2 + self.location[1]
                ),
                int(-self.visible_x / 2 + self.location[0]) : int(
                    self.visible_x / 2 + self.location[0]
                ),
            ] = 0
        return score

    def step(self, action, DISPLAY=False):
        """
        defining navigation:
        let's assume each step takes 1 second and moves the agent for =1 (s) * V (m/s)
        the idea is that action is the absolute velocity. now if you have a heavy wind the cost will be higher
        but the absolute velocity won't change.
        """
        self.action = action
        self.reward = 0

        if self.action_mode == "cont":
            # new location is assigned and reward of movement is calculated (move_cost())
            self.move_by_velocity()
        if self.action_mode == "disc":
            self.move_by_tile()
        self.path.append(self.location)
        info = {
            "location": self.location,
            "boundaries": self.boundaries,
            "scale_x": self.visible_x / self.cfg.FRAME_W,
            "scale_y": self.visible_y / self.cfg.FRAME_H,
            "step_count": self.step_count,
            "battery_levels": self.battery,
        }
        self.info_list.append(info)

        if self.cfg.show_location:
            logging.info(f"location after step {info}")

        if self.battery < 1:
            self.reward -= 10
            self.done = True
            self.close()

        if self.render == True and self.done == False:
            self.renderer()
            time.sleep(self.cfg.sleep_time) if self.cfg.sleep_time > 0 else None

        # exit()
        # _score = self.fetch_anomaly()
        _score=0
        # if _score>0:
        #     print('step',self.step_count, '\n this reward: ', _score, '\n')
        self.reward += _score
        self.total_reward += self.reward
        self.step_count += 1
        # print('STEP MTHD, count: ', self.step_count)

        # if self.fetch_anomaly()>0:
        #     print('step',self.step_count, '\n this reward: ', self.reward, '\n')
        #     print('Total rewards is:', self.total_reward)

        self.reward = float(self.reward)

        ### defining observation
        if self.observation_mode == "cont":
            # making sure the image values are normalized
            observation = self.fetch_frame()
            observation = np.array(observation, dtype=np.uint8)
            observation = observation.reshape((self.cfg.FRAME_H, self.cfg.FRAME_W, 3))
        else:
            observation = [
                self.location[0],
                self.location[1],
                self.location[2],
                self.battery,
                self.wind[0],
                self.wind[1],
            ]
            observation = np.array(observation, dtype=np.float16)

        if DISPLAY == True:
            self.display_info()
        if self.cfg.MAX_STEPS < self.step_count:
            self.done = True
        # print('step count: ', self.step_count)
        # print(' x size on world: ', self.visible_x)
        # print( 'y size on world: ', self.visible_y)
        return (
            observation,
            self.reward,
            self.done,
            self.done,
            info,
        )  # now needs both terminated and turncated booleans, passed done for both.

    def move_by_velocity(self):
        self.abs_velocity = self.action

        if self.abs_velocity[0] < 0:
            new_x = max(self.location[0] + self.abs_velocity[0], self.cfg.WORLD_XS[0])
        else:
            new_x = min(self.location[0] + self.abs_velocity[0], self.cfg.WORLD_XS[1])

        if self.abs_velocity[1] < 0:
            new_y = max(self.location[1] + self.abs_velocity[1], self.cfg.WORLD_YS[0])
        else:
            new_y = min(self.location[1] + self.abs_velocity[1], self.cfg.WORLD_YS[1])

        if self.abs_velocity[2] < 0:
            new_z = max(self.location[2] + self.abs_velocity[2], self.cfg.WORLD_ZS[0])
        else:
            new_z = min(self.location[2] + self.abs_velocity[2], self.cfg.WORLD_ZS[1])

        self.location = (new_x, new_y, new_z)
        self.reward = -self.move_cost()

    ###############################################################################
    def reset(self, **kwargs):
        ### for COARSE Coding there is an auxiliary function to get location from state
        # print('\n \n \n \n  reset happened!!! \n \n \n \n')
        print("number of active threads:", threading.active_count())
        self.close()

        self.done = False
        time.sleep(0.01)
        self.thread = Thread(target=self.update_frame, args=(), daemon=True)
        self.thread.start()

        # self.location = self.cfg.init_location
        self.location = self.cfg.init_location.copy()

        self.world = (
            self.world_genertor()
            if not self.cfg.load_from_geotiff
            else self.load_from_geotiff()
        )
        self.battery = self.cfg.FULL_BATTERY  # [x,y,z,] m
        self.reward = 0
        self.total_reward = 0
        self.step_count = 0

        self.prev_reward = 0
        self.score = 0

        ### defining observation
        if self.observation_mode == "cont":
            observation = self.fetch_frame()
            observation = np.array(observation, dtype=np.uint8)
            observation = observation.reshape(
                (self.cfg.FRAME_H, self.cfg.FRAME_W + 1, 1)
            )
        else:
            observation = [
                self.location[0],
                self.location[1],
                self.location[2],
                self.battery,
                self.wind[0],
                self.wind[1],
            ]
            observation = np.array(observation, dtype=np.float16)
        info = {}

        # for training we need two returns, why?!!!
        # return observation, info

        # for testing we need one return, why?!!!
        return np.array(observation, dtype=np.float16), info

    def load_geotiff(self):
        geo = cv2.imread(self.image_path)
        logging.info(
            f"geotiff loaded from file {self.image_path} with size {geo.shape[0], geo.shape[1]} and padding of {self.cfg.PADDING}"
        )

        self.cfg.wolrd_size_including_padding = [geo.shape[1], geo.shape[0]]

        self.cfg.WORLD_XS = [
            self.cfg.PADDING,
            self.cfg.wolrd_size_including_padding[0] - self.cfg.PADDING,
        ]
        self.cfg.WORLD_YS = [
            self.cfg.PADDING,
            self.cfg.wolrd_size_including_padding[1] - self.cfg.PADDING,
        ]
        logging.info(
            f"Minimum and maximum bounds for Xs of the world: {self.cfg.WORLD_XS}"
        )
        logging.info(
            f"Minimum and maximum bounds for Ys of the world: {self.cfg.WORLD_YS}"
        )
        logging.info(
            f"Minimum and maximum bounds for Zs of the world: {self.cfg.WORLD_ZS}"
        )
        return geo

    def padd_based_on_resolution(self, img):
        """
        This method is going to padd all four edges of the image such that the maximum
        elevation will never cause the drone view to attmept to look outside the image
        """
        pass

    def find_visible_boundaries(self, location):
        pass

    def world_genertor(self, write_to_file=True):
        ### the padded area of the world is were the drone cannot go to but may appear in the frame
        seeds = self.cfg.SEEDS
        height = int(self.cfg.wolrd_size_including_padding[1])
        width = int(self.cfg.wolrd_size_including_padding[0])
        size = (height, width)
        world = np.zeros(size, dtype=int)
        logging.info(
            f"world created including the padded borders with size:{world.shape}"
        )
        square_corners = []
        for s in range(0, seeds):
            ### corner of each square corner=[x,y]
            corner = [
                random.randint(self.cfg.WORLD_XS[0], self.cfg.WORLD_XS[1]),
                random.randint(self.cfg.WORLD_YS[0], self.cfg.WORLD_YS[1]),
            ]
            ### list of all square corners
            square_corners.append(corner)
            square_size = random.randint(
                self.cfg.square_size_range[0], self.cfg.square_size_range[1]
            )
            for i in range(0, square_size):
                for j in range(0, square_size):
                    try:
                        world[corner[1] + j][corner[0] + i] = 1
                    except:
                        pass

        world_img = np.uint8((1 - world) * 255)
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"images/world_{timestamp}_seed{seeds}.png"
            cv2.imwrite(filename, world_img)
            logging.info(f"World image saved as {filename}")
        except Exception as e:
            logging.error(f"Failed to save world image: {e}")

        return world

    def loc_from_state(self):
        state_x_size = (self.cfg.WORLD_XS[1] - self.cfg.WORLD_XS[0]) / self.cfg.STATES_X
        state_y_size = (self.cfg.WORLD_YS[1] - self.cfg.WORLD_YS[0]) / self.cfg.STATES_Y
        state_z_size = (self.cfg.WORLD_YS[1] - self.cfg.WORLD_YS[0]) / self.cfg.STATES_Z
        loc = [
            self.cfg.WORLD_XS[0] + state_x_size * (self.state[0] - 1),
            self.cfg.WORLD_YS[0] + state_y_size * (self.state[1] - 1),
            self.cfg.WORLD_ZS[0] + state_z_size * (self.state[2] - 1),
        ]
        ### returns a location=[x,y,z] where states=[1,1,1] corresponds to loc=[100,100,60] meteres
        return loc

    def close(self):
        # print('trying to close the env and destroy windows...')
        logging.info("Closing the env")
        self.done = True
        time.sleep(0.1)
        self.imager_thread_name.join()
        cv2.destroyAllWindows()
        self.dump_step_history_json()
        if self.cfg.save_map_to_file and self.cfg.create_explored_map:
            self.map_of_detections()
            self.show_write_explored_map()



    def dump_step_history_json(self, filename=None):
        # saving the list of info with time stamp
        if filename == None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"flight_info_{timestamp}.json"
        #checking if flight_logs/ exists then dumpint there
        if not os.path.exists('flight_logs'):
            os.makedirs('flight_logs')
        filename = os.path.join('flight_logs',filename)
        logging.info(f"Saving flight info to {filename}")
        # dumping the json file
        with open(filename, "w") as f:
            json.dump(self.info_list, f)

    def show_write_explored_map(self):
        logging.info(
            f"Adding the path to image and saving to file. Path length: {len(self.path)}"
        )
        for i in range(1, len(self.path)):
            self.explored_map = cv2.line(
                self.explored_map,
                tuple(self.path[i - 1][:2]),
                tuple(self.path[i][:2]),
                (255, 255, 0),
                3,
            )
            if self.path[i - 1][2] > self.path[i][2]:
                self.explored_map = cv2.circle(
                    self.explored_map, tuple(self.path[i][:2]), 5, (255, 0, 0), -1)
            elif self.path[i - 1][2] < self.path[i][2]:
                self.explored_map = cv2.circle(
                    self.explored_map, tuple(self.path[i][:2]), 5, (0, 255, 0), -1)
        output_name = self.image_path.split(".")[0] + "_path.png"
        # showing the output image for 5 seconds before saving to file
        cv2.imshow(" Resulted Path", self.explored_map)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        cv2.imwrite(output_name, self.explored_map)
    
    def map_of_detections(self):
        for dict in self.info_list:
            if len(dict["detections"]) > 0:
                
                    x1, y1, x2, y2 = dict["detections"][:4]
                    X1= x1 +  dict["boundaries"][2]
                    Y1= y1 +  dict["boundaries"][0]
                    X2= x2 +  dict["boundaries"][2]
                    Y2= y2 +  dict["boundaries"][0]
                    self.explored_map = cv2.rectangle(
                        self.explored_map, (X1, Y1), (X2, Y2), (0, 0, 255), 2
                    )
        output_name = self.image_path.split(".")[0] + "_detections.png"
        cv2.imwrite(output_name, self.explored_map)
        logging.info(f"Map of detections saved to {output_name}")



        
    ### method receives frame as np array adds a column the end that represent battery level
    def concat_battery(self, input_frame):
        full_pixels_count = int(self.cfg.FRAME_H * self.battery / 100)
        full_pixels = np.zeros([full_pixels_count, 1])
        full_pixels.astype(int)
        empty_pixels = (np.zeros([self.cfg.FRAME_H - full_pixels_count, 1]) + 1) * 255
        empty_pixels.astype(int)

        battery_img = np.uint8(np.concatenate((empty_pixels, full_pixels)))
        self.output_frame = np.concatenate((input_frame, battery_img), axis=1)

        return self.output_frame

    def move_cost(self):
        # finding the relative angle of wind to drone
        # print('inside move_cost() method \n actions: ', self.action[0] , self.action[1], 'winds: ', self.wind[0], self.wind[1])
        try:
            self.wind_angle = degrees(
                acos(
                    (self.action[0] * self.wind[0] + self.action[1] * self.wind[1])
                    / (
                        sqrt(self.action[0] ** 2 + self.action[1] ** 2)
                        * sqrt(self.wind[0] ** 2 + self.wind[1] ** 2)
                    )
                )
            )
        except:
            # logging.error('wind wind angle is zero')
            self.wind_angle = 0
        # finding the relative velocity of wind to drone in absolute value
        self.relative_velocity = sqrt(
            (self.action[0] - self.wind[0]) ** 2 + (self.action[1] - self.wind[1]) ** 2
        )

        try:
            self.drag = (
                self.cfg.drag_table.iloc[
                    round(self.wind_angle / 45.0),
                    round((self.relative_velocity - 12) / 5),
                ]
                * self.drag_normalizer_coef
            )
        except:
            print("relative velocity/angles out of bounds.")
            # defualt drag
            self.drag = 0.1 * self.drag_normalizer_coef
        cost = self.drag * self.relative_velocity**2

        ### method to find the step cost based on drag force, for now everything costs 1
        # self.cost= self.c_d *((self.action[0]-self.wind[0])**2 + (self.action[1]-self.wind[1])**2)

        if self.battery_inloop == True:
            self.battery = max(0, self.battery - cost)
            return cost
        else:
            # print(self.battery)
            return 0

    def renderer(self):
        try:
            ##drone crop
            drone_crop = self.fetch_frame()
            cv2.imshow("Drone View", drone_crop)

            ##world crop
            img = self.world_img.copy()
            img = cv2.rectangle(
                img,
                (self.boundaries[2], self.boundaries[0]),
                (self.boundaries[3], self.boundaries[1]),
                (255, 0, 0),
                5,
            )

            # resize such that it fits the screen maintaining the aspect ratio
            AR = self.world_img.shape[1] / self.world_img.shape[0]
            long_edge = 1000
            short_edge = int(long_edge / AR)
            img_resized = cv2.resize(img, (long_edge, short_edge))

            ### adding text
            img_resized = cv2.putText(
                img_resized,
                "East WIND: "
                + str(np.round(-self.wind[0], 2))
                + " North WIND:"
                + str(np.round(self.wind[1], 2)),
                (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            img_resized = cv2.putText(
                img_resized,
                "step " + str(self.step_count),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            img_resized = cv2.putText(
                img_resized,
                "battery: " + str(np.round(self.battery, 2)),
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            img_resized = cv2.putText(
                img_resized,
                "Heading angle w.r.t wind: " + str(np.round(self.wind_angle, 2)),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            img_resized = cv2.putText(
                img_resized,
                "flight altitude: " + str(np.round(self.location[2], 2)),
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("World view", img_resized)

            if self.cfg.create_explored_map:
                explored_map_resized = cv2.resize(
                    self.explored_map, (long_edge, short_edge)
                )
                cv2.imshow("Explored map", explored_map_resized)

        except Exception as e:
            logging.info(f"Frame not available for render! with error {e}")

        if cv2.waitKey(1) == ord("q"):
            print("Q hit:")
            self.done = True
            self.close()

    def display_info(self):
        print("==== INFO ==== \n")
        # print('step count: ', self.step_count)
        # print('battery level: ', self.battery)
        # print('current location: ', self.location)
        # print('current wind angle: ', self.wind_angle)
        # print('relative velocity: ', self.relative_velocity)
        print("step:", self.step_count, " reward: ", self.reward)
        print("\n total reward: ", self.total_reward)
        print("========= \n")

    def _find_visible_boundaries(location):
        """
        This method is used to find the visible boundaries of the drone view based on the location
        of the drone and the field of view of the camera.
        This method is meant to be accessible outside the class.
        """
        pass
