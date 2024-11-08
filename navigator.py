from drone_environment import droneEnv
import time
import numpy as np
import cv2  # Add this import for keyboard input

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

    def navigate(self):
        for i in range(1):
            print('Iteration: ', i, '\n supposed location: ', self.env.location, 'configurations: ', self.env.cfg.init_location)
            while True:
                if self.LTR == 1:
                    while self.env.done == False and abs(self.env.location[0] - self.env.cfg.WORLD_XS[1]) > 1:
                        obs, reward, done, _, info = self.env.step([self.step_x, 0, 0])
                        self.steps += 1
                        self.rewards.append(reward)
                        yield obs
                if self.LTR == -1:
                    while self.env.done == False and abs(self.env.location[0] - self.env.cfg.WORLD_XS[0]) > 1:
                        obs, reward, done, _, info = self.env.step([self.step_x, 0, 0])
                        self.steps += 1
                        self.rewards.append(reward)
                        yield obs
                self.step_x = -self.step_x
                self.LTR = -self.LTR
                if self.env.done == False and abs(self.env.location[1] - self.env.cfg.WORLD_YS[1]) > 1:
                    obs, reward, done, _, info = self.env.step([0, self.step_y, 0])
                    yield obs
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
                print('locations', info)
                i = 0
                yield obs
            if action != [0, 0, 0]:
                obs, reward, done, _, info = self.env.step(action)
                print(info)
                yield obs
        self.env.close()
