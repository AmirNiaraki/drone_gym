from drone_environment import droneEnv
import time
import cv2
import numpy as np




env = droneEnv(observation_mode='cont', action_mode='cont', render=True)

# Function to update the rectangle's position
def keyboard_stepper(key):
    x,y,z=0,0,0
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
    action=[x,y,z]

    return action

def main():
    i=1
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Esc key to break
            break

        action=keyboard_stepper(key)
       
        if i==1:
            obs, reward, done, _, info =env.step(action)
            print(info)
            i=0
        if action != [0,0,0]:
            # print(action)

            obs, reward, done, _, info =env.step(action)
            print(info)
        time.sleep(0.1)
    

if __name__ == '__main__':
    main()
    env.close()