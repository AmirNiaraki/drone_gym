# run with python3 image_editor.py filename.png

import cv2
import numpy as np
import argparse

# Create a mask for the selected region
def create_mask(image, points_list):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for points in points_list:
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

# Draw lines between consecutive points with red color for real-time display
def draw_lines(image, points):
    temp_image = image.copy()
    for i in range(len(points) - 1):
        cv2.line(temp_image, points[i], points[i + 1], (0, 0, 255), 2)  # Red color (BGR)
    return temp_image

# Turn pixels outside the points black
def apply_mask(image, mask):
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Select and crop a region of interest in an image")
parser.add_argument("image_path", type=str, help="Path to the input image")

args = parser.parse_args()

# Load the image in color
image = cv2.imread(args.image_path)

# Invalid image name/path
if image is None:
    print("Error: Unable to load the image.")
    exit(1)

# Convert the image to grayscale for masking
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize a list to store selected points for each polygon
all_points = []
current_polygon_points = []

# Callback function for placing points
def select_point(event, x, y, flag, param):
    global current_polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon_points.append((x, y))
        temp_image = draw_lines(image, current_polygon_points)  # Draw red lines for real-time display
        for points in all_points:
            temp_image = draw_lines(temp_image, points)
        cv2.imshow('Select Region', temp_image)

# create a window for displaying the image to the user
cv2.namedWindow('Select Region')
cv2.imshow('Select Region', image)
cv2.setMouseCallback('Select Region', select_point)

while True:
    key = cv2.waitKey(1)

    # Press 'Enter' to complete the current polygon and start a new one
    if key == 13:  
        if len(current_polygon_points) > 1:
            current_polygon_points.append(current_polygon_points[0])  # Connect the last point to the first point
            all_points.append(current_polygon_points)
            current_polygon_points = []

    # Press 'q' to darken pixels outside each polygon and save the image
    elif key == ord('q'):
        break

# Create and apply the final mask for all polygons
final_mask = create_mask(gray_image, all_points)
result = apply_mask(image, final_mask)

# if the image comes out as RGB, convert to Grayscale
try:
    result = np.amax(np.array(result, dtype=int), axis=2)
except:
    pass

# Clean up the window, write the image to file
cv2.destroyAllWindows()
cv2.imwrite('output_image.png', result)
# Div by 255 to normalize (black = 1, white = 0)
# This is necessary for the environment to properly upload
np.save("output_image", result // 255)
np.save("output_image_points", all_points)
