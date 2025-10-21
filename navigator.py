from drone_environment import droneEnv
import time
import numpy as np
import cv2  # Add this import for keyboard input
import logging
import math
class Navigator:
    def __init__(self, env):
        self.env = env

    def navigate(self):
        raise NotImplementedError("Subclasses should implement this!")


class HierarchicalNavigator(Navigator):
    def __init__(self, env):
        super().__init__(env)
        self.phase='sampling' # sampling and CCPP (Complete Coverage Path Planning)
        self.edge_discretization_segments = 5 #number of devides per edge (so 2 for parameter means the image is divided into 4)
        self.sampling_velocity=10 #m/s
        self.score_per_step_dict = []

    def generate_planar_sampling_waypoints(self):
        # Generate (x,y) waypoints for sampling phase
        logging.info(f'Deviding the world into {self.edge_discretization_segments*self.edge_discretization_segments} segments ')
        segment_width=self.env.cfg.wolrd_size_including_padding[0]/self.edge_discretization_segments
        segment_height=self.env.cfg.wolrd_size_including_padding[1]/self.edge_discretization_segments
        logging.info(f'segment width: {segment_width}, segment height: {segment_height}')
        waypoints=[]
        segment_lines_x=[]
        segment_lines_y=[]
        for i in range(self.edge_discretization_segments):
            for j in range(self.edge_discretization_segments):
                waypoints.append((j*segment_width+segment_width/2,i*segment_height+segment_height/2))
                segment_lines_x.append(j*segment_width)
                segment_lines_y.append(i*segment_height)
        if self.env.cfg.save_map_to_file:
            self.env.write_segments(segment_lines_x, segment_lines_y, self.edge_discretization_segments*self.edge_discretization_segments)
        return waypoints

    def generate_3d_sampling_waypoints(self, waypoints_2d):
        # Generate (x,y,z) waypoints for sampling phase
        waypoints_3d=[]
        height_fraction_counts = len(waypoints_2d)
        logging.info(f'sampling at {height_fraction_counts} different heights')
        height_interval = int((self.env.cfg.WORLD_ZS[1]-self.env.cfg.WORLD_ZS[0])/height_fraction_counts)
        for i in range(height_fraction_counts):
            waypoints_3d.append((waypoints_2d[i][0],waypoints_2d[i][1],self.env.cfg.WORLD_ZS[0]+i*height_interval))
        logging.info(f'waypoints_3d: {waypoints_3d}')
        return waypoints_3d , height_interval          
        
    def calculate_distance(self, point1, point2):
        '''
        The points given in the form of (x,y,z)
        the distance is euclidean distance as scalar
        '''
        distance=math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
        # return np.linalg.norm(np.array(point1)-np.array(point2))
        return distance

    def find_direction(self, point1, point2):
        '''
        The points given in the form of (x,y,z)
        the direction is a unit vector pointing from point1 to point2 in 2D
        '''
        distance=abs(self.calculate_distance(point1,point2))
        dir_x=(point2[0]-point1[0])/distance
        dir_y=(point2[1]-point1[1])/distance
        unit_vector=[dir_x,dir_y,0]
        return unit_vector

    def navigate(self):
        
        if self.phase == 'sampling':
            waypoints_2d=self.generate_planar_sampling_waypoints()
            way_points_3d, height_interval=self.generate_3d_sampling_waypoints(waypoints_2d)

            self.env.location = (self.env.cfg.init_location[0], self.env.cfg.init_location[1], self.env.cfg.WORLD_ZS[0])
            for i in range(len(way_points_3d)):
                # self.score_per_step_dict['height'] = self.env.location[2]
                # self.score_per_step_dict['start_step'] = self.env.step_count
                step_info = {
                    'height': self.env.location[2],
                    'start_step': self.env.step_count
                }
                logging.info(f'current location: {self.env.location} going to wardd way point at {way_points_3d[i]}')


                ### navigation peice from p1 to p2
                direction=self.find_direction(self.env.location,way_points_3d[i])
                logging.info(f'direction: {direction}')
                # time.sleep(1)

                while True and self.env.done==False:
                    distance=self.calculate_distance(self.env.location,way_points_3d[i])
                    # logging.info(f'distance: {distance}')
                    if int(distance)<=10:
                        break
                    else:
                        # move towards the waypoint
                        x=direction[0]*self.sampling_velocity
                        y=direction[1]*self.sampling_velocity
                        z=0
                        obs, reward, done, _, info = self.env.step([x, y, z])
                        # logging.info(f'location after step {info}')
                        yield obs, info
                ### end of navigation starting postprocessing the outcome of the p1 to p2 navigation
                step_info['end_step'] = self.env.step_count
                self.calculate_score_per_step(step_info)
                logging.info(f'waypoint number {i} reached, changing height by {height_interval}')
                obs, reward, done, _, info = self.env.step([0, 0, height_interval])
            logging.info(f'All waypoints reached; end of sampling. Finding the optimum altitude: {self.score_per_step_dict}')

            self.env.close()

        elif self.phase == 'CCPP':
            pass
        else:
            raise ValueError('Invalid phase')
    
    def calculate_score_per_step(self,step_info):
        
        start_step = step_info.get('start_step')
        end_step = step_info.get('end_step')
        average_score = 0
        if start_step is not None and end_step is not None:
            total_score = 0
            step_count = 0
            for info in self.env.info_list:
                if start_step <= info['step_count'] <= end_step:
                    total_score += info.get('detection_score')
                    step_count += 1
            average_score = total_score / step_count if step_count > 0 else 0
            step_info['average_score'] = average_score
            logging.info(f'Average score between steps {start_step} and {end_step} at height {step_info['height']}: {average_score}')
            self.score_per_step_dict.append(step_info)

class CompleteCoverageNavigator(Navigator):
    def __init__(self, env):
        super().__init__(env)
        self.step_x = 100
        self.step_y = 100
        self.LTR = 1
        self.steps = 0
        self.rewards = []
        # setting the height to maximum in self.location which is a tuple
        self.env.location = (self.env.cfg.init_location[0], self.env.cfg.init_location[1], self.env.cfg.WORLD_ZS[1])


    def navigate(self):
        first_step = False
        for i in range(1):
            print('Iteration: ', i, '\n supposed location: ', self.env.location, 'configurations: ', self.env.cfg.init_location)
            
            while True:

                if self.LTR == 1:
                    while self.env.done == False and abs(self.env.location[0] - self.env.cfg.WORLD_XS[1]) > 1:
                        obs, reward, done, _, info = self.env.step([self.step_x, 0, 0])
                        if first_step:
                            time.sleep(10)
                            first_step = False
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
        self.step_size=100

    def keyboard_stepper(self, key):
        x, y, z = 0, 0, 0
        if key == ord('w'):  # Up
            y -= self.step_size
        elif key == ord('s'):  # Down
            y += self.step_size
        elif key == ord('a'):  # Left
            x -= self.step_size
        elif key == ord('d'):  # Right
            x += self.step_size
        elif key == ord('z'):
            z -= self.step_size
        elif key == ord('x'):
            z += self.step_size
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
