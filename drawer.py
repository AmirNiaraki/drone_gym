"""
This python code describes the WorldDrawer class which inherits droneEnv and uses opencv
to first open a window with the size of the world as it is constructed with world_generator() method,
and provides the functionality to use right click of a mouse to generate black pixels on the large
white picture that is the world.
"""

import cv2
import numpy as np
from IPP_drone_path_planner import droneEnv

class WorldDrawer(droneEnv):
    def __init__(self):
        super().__init__('cont', render=True)
        print(type(self.cfg.WORLD_YS[1]+self.cfg.PADDING), type(self.cfg.WORLD_YS[1]), type(self.cfg.PADDING))
        self.size=(int(self.cfg.WORLD_YS[1]+self.cfg.PADDING),int(self.cfg.WORLD_XS[1]+self.cfg.PADDING))
        print(type(self.cfg.PADDING))
        self.world=np.zeros(self.size, dtype=int)
        self.world_img=np.uint8((1-self.world)*255)
        self.black_list=[]
        self.window_name = 'World Drawer'
        cv2.namedWindow(self.window_name)
        self.thickness=5



    def show_world(self):
        cv2.imshow(self.window_name, self.world_img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
        cv2.destroyAllWindows()


    def draw_pixel(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.black_list.append((x,y))
            self.world_img[y, x] = 0

            # cv2.imshow(self.window_name, self.world_img)

    def run(self):
        print('wait for right clicks to generate black pixels on the white world')
        print('press q to finish')
        while True:
            cv2.setMouseCallback(self.window_name, self.draw_pixel)
            cv2.imshow(self.window_name, self.world_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cv2.destroyAllWindows
    
        for i in range(len(self.black_list)):
            self.world[self.black_list[i][1]-self.thickness:self.black_list[i][1]+self.thickness,
                       self.black_list[i][0]-self.thickness:self.black_list[i][0]+self.thickness]=1
        self.world_img=np.uint8((1-self.world)*255)
        self.show_world()

    def write_to_file(self):
        cv2.imwrite('drawn_world.png', self.world_img)
        np.save('drawn_world', self.world)
        # self.world=np.load('test_world.npy')
        print('world saved to file')
        self.calulcate_all_black_pixles()

    def calulcate_all_black_pixles(self):
        print(self.world.shape)

        print('number of all black pixels are: ', np.sum(self.world))


if __name__ == '__main__':
    wd = WorldDrawer()
    wd.run()
    wd.write_to_file()
