from drone_environment import droneEnv
import time
import numpy as np
import cv2  # Add this import for keyboard input
import logging
class Navigator:
    def __init__(self, env):
        self.env = env

    def navigate(self):
        raise NotImplementedError("Subclasses should implement this!")

class CompleteCoverageNavigator(Navigator):
    def __init__(self, env):
        super().__init__(env)
        self.step_x = 20
        self.step_y = 20
        self.LTR = 1
        self.steps = 0
        self.rewards = []
        # setting the height to maximum in self.location which is a tuple
        self.env.location = (self.env.cfg.init_location[0], self.env.cfg.init_location[1], self.env.cfg.WORLD_ZS[1])


    def navigate(self):
        for i in range(1):
            print('Iteration: ', i, '\n supposed location: ', self.env.location, 'configurations: ', self.env.cfg.init_location)
            while True:
                if self.LTR == 1:
                    while self.env.done == False and abs(self.env.location[0] - self.env.cfg.WORLD_XS[1]) > 1:
                        obs, reward, done, _, info = self.env.step([self.step_x, 0, 0])
                        self.steps += 1
                        self.rewards.append(reward)
                        yield obs, info
                if self.LTR == -1:
                    while self.env.done == False and abs(self.env.location[0] - self.env.cfg.WORLD_XS[0]) > 1:
                        obs, reward, done, _, info = self.env.step([self.step_x, 0, 0])
                        self.steps += 1
                        self.rewards.append(reward)
                        yield obs, info
                self.step_x = -self.step_x
                self.LTR = -self.LTR
                if self.env.done == False and abs(self.env.location[1] - self.env.cfg.WORLD_YS[1]) > self.env.cfg.PADDING+1:
                    obs, reward, done, _, info = self.env.step([0, self.step_y, 0])
                    yield obs, info
                else:
                    break
            print(f'length of rewards: {len(self.rewards)}', f'number of steps: {self.steps}', 'total rewards: ', sum(self.rewards))
        self.env.close()

class KeyboardNavigator(Navigator):
    def __init__(self, env):
        super().__init__(env)

    def keyboard_stepper(self, key):
        x, y, z = 0, 0, 0
        if key == ord('w'):  # Up
            y -= 10
        elif key == ord('s'):  # Down
            y += 10
        elif key == ord('a'):  # Left
            x -= 10
        elif key == ord('d'):  # Right
            x += 10
        elif key == ord('z'):
            z -= 10
        elif key == ord('x'):
            z += 10
        elif key == ord('o'):
            x, y, z = 1000,1000,1000
        action = [x, y, z]
        return action

    def navigate(self):
        i = 1
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Esc key to break
                break
            action = self.keyboard_stepper(key)
            if i == 1:
                obs, reward, done, _, info = self.env.step(action)

                i = 0
                yield obs, info
            if action != [0, 0, 0] and action != [1000,1000,1000]:
                obs, reward, done, _, info = self.env.step(action)
                logging.info(f'location after step {info}')
                yield obs, info
            if action == [1000,1000,1000]:
                info = self.env.info_list[-1]  # Retrieve the last info from the list
                x1, y1, x2, y2 = 10, 10, 20, 20  # this is a dummy detection as placeholder for bbox
                info['detections'] = [x1, y1, x2, y2]
                logging.info(f'created dummy detection at: {info["detections"]}')
                self.env.info_list[-1] = info  # Update the last info in the list

                yield obs, info
        self.env.close()
