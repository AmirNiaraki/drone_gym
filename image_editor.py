#run with 'python select_region.py your_image.jpg'

import cv2
import numpy as np
import argparse

# Create a mask for the selected region
def create_mask(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

# Turn pixels outside the points black
def apply_mask(image, mask):
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Select and crop a region of interest in an image")
parser.add_argument("image_path", type=str, help="Path to the input image")

args = parser.parse_args()

# Load the image in grayscale
image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

# Invalid image name/path
if image is None:
    print("Error: Unable to load the image.")
    exit(1)

# Initialize the list to store selected points
points = []

# Callback function for placing points
def select_point(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        mask = create_mask(image, points)
        global result
        result = apply_mask(image, mask)
        # cv2.imshow('Result', result)

# create a window for displaying the image to the user
cv2.namedWindow('Select Region')
cv2.imshow('Select Region', image)
cv2.setMouseCallback('Select Region', select_point)

#wait until the enter key is pressed to darken pixels outside of region
while True:
    key = cv2.waitKey(1)
    if key == 13:
        break

#clean up the window, write the image to file
cv2.destroyAllWindows()
cv2.imwrite('output_image.png', result)
# div by 255 to normalize (black = 1, white = 0)
# this is necessary for the environment to properly upload
np.save("output_image", result / 255)
np.save("output_image_points", points)