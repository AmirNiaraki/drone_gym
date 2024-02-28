import cv2
import numpy as np
import argparse
import os

def interpolate(image):
    global all_points

    polygons = []

    for polygon in all_points:
        points = []

        for i in range(len(polygon) - 1):
            x_start, y_start = polygon[i]
            x_end, y_end = polygon[i + 1]

            # Bresenham's line algorithm
            edge = list(zip(*cv2.line(np.zeros_like(image), 
                        (x_start, y_start), (x_end, y_end), 
                        color=255, thickness=1)[0].nonzero())
                    )
            points.extend(edge)
        
        polygons.append(points)

    return polygons

def create_mask(image, points_list):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for points in points_list:
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

def apply_mask(image, mask):
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def draw_lines(image, points):
    temp_image = image.copy()
    for i in range(len(points) - 1):
        cv2.line(temp_image, points[i], points[i + 1], (0, 255, 255), 3)
    return temp_image

def find_box(points_list, margin):
    all_x = [point[0] for points in points_list for point in points]
    all_y = [point[1] for points in points_list for point in points]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add a margin to the bounding box
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    return int(min_x), int(min_y), int(max_x), int(max_y)

def resize_image(image, factor):
    return cv2.resize(image, (int(image.shape[1] / factor), int(image.shape[0] / factor)))

def convert_to_fullsize(factor):
    global all_points
    global current_polygon_points

    resized_all_points = []
    resized_current_polygon = []

    for polygon in all_points:
        resized_polygon = [(int((x * factor)), int((y * factor))) for x, y in polygon]
        resized_all_points.append(resized_polygon)

    resized_current_polygon = [(int((x * factor)), int((y * factor))) for x, y in current_polygon_points]

    all_points = resized_all_points
    current_polygon_points = resized_current_polygon

def callback(event, x, y, flags, param):
    global current_polygon_points

    # if left ctrl and left click is pressed together
    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
        # add a point
        current_polygon_points.append((x, y))

def erase():
    global all_points
    global current_polygon_points

    # If we are currently in progress of drawing a polygon...
    if current_polygon_points:
        # Erase the last line in the current polygon
        current_polygon_points.pop()
    # Else if we haven't started a new polygon...
    elif all_points:
        # Pop the last added polygon off, set to current polygon
        current_polygon_points = all_points.pop()
        # Erase the last line of that polygon
        current_polygon_points.pop()

    return current_polygon_points, all_points

def refresh(resized):
    global all_points
    global current_polygon_points

    # draw the lines on the new image
    for points in all_points:
        resized = draw_lines(resized, points)

    if current_polygon_points:
        resized = draw_lines(resized, current_polygon_points)

    cv2.imshow('Select Region', resized)

    return resized

def initialize(image_path, factor):
    # Save the extension for when we save the image later
    _, file_extension = os.path.splitext(image_path)

    # Load the original image in color
    full_size = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Handle invalid image name/path
    if full_size is None:
        print("Error: Unable to load the image.")
        exit(1)
    
    # Downsize image by a factor
    resized = resize_image(full_size, factor)

    # If the file extension is not ".tif", convert to grayscale
    if file_extension != ".tif":
        # Convert the image to grayscale for masking
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # .tif files color channel as RGB, convert to CV2's BGR
    #else:
        #full_size = cv2.cvtColor(full_size, cv2.COLOR_RGB2BGR)
        #resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    # Initialize window size and position
    # 800 x 600 seems okay for now
    window_width, window_height = 800, 600
    cv2.namedWindow('Select Region', cv2.WINDOW_NORMAL)
    # Resize cv2 window so it's not outside of our physical screen
    cv2.resizeWindow('Select Region', window_width, window_height)
    # Set window in the top-left corner of the image if larger than the window
    cv2.moveWindow('Select Region', 0, 0)

    # Set callbacks
    cv2.setMouseCallback('Select Region', callback)

    return file_extension, full_size, resized

def complete_polygon():
    global current_polygon_points
    global all_points

    # As long as you have more than 3 points...
    if len(current_polygon_points) > 3:
        # Connect the last point to the first, completing the polygon
        current_polygon_points.append(current_polygon_points[0])
        # Append the polygon to the list of all polygons
        all_points.append(current_polygon_points)
        # Clear the current polygon
        current_polygon_points = []
    
    return current_polygon_points, all_points

def write(full_size, file_extention, factor):
    global all_points

    # convert the polygons from the resized values to the full sized
    convert_to_fullsize(factor)

    # Create and apply the final mask for all polygons
    final_mask = create_mask(full_size, all_points)
    result = apply_mask(full_size, final_mask)

    print(all_points)

    # find the bounding box around the polygons with a margin of 10 and crop
    min_x, min_y, max_x, max_y = find_box(all_points, 10)
    result = result[min_y:max_y, min_x:max_x]

    if file_extention != ".tif":
        # if the image comes out as RGB, convert to Grayscale
        try:
            result = np.amax(np.array(result, dtype=int), axis=2)
        except:
            pass

    # Clean up the window, write the image to file
    cv2.destroyAllWindows()
    output = 'output_image' + file_extention
    cv2.imwrite(output, result)
    # Div by 255 to normalize (black = 1, white = 0)
    # This is necessary for the environment to properly upload
    np.save("output_image", result // 255)
    # np.save("output_polygons", interpolate(result))

def process(image_path):
    global all_points
    global current_polygon_points

    # set image resize by factor
    factor = 10

    # List of all polygons which is an array of the 
    # sets of points that make one polygon 
    all_points = []
    # Coordinate points that make up the current polygon we are working on
    current_polygon_points = []

    # load image, resize, set window, preserve file extention, set click callback
    file_extension, full_size, resized = initialize(image_path, factor)


    # Millisecond loop to read user keyboard input
    while True:
        # Get key event every millisecond
        key = cv2.waitKey(1)

        # Press 'Enter' to complete the current polygon and start a new one
        if key == 13:  
            complete_polygon()

        # Press 'e' to erase the last line
        elif key == ord('e'):
            erase()

            # reset the resized image and redraw all lines
            resized = refresh(resize_image(full_size, factor))

        # Press 'q' to mask outside each polygon and save the image
        elif key == ord('q'):
            break

        # refresh display
        resized = refresh(resized)

    # mask original, saved outputs, clean up environment
    write(full_size, file_extension, factor)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and crop a region of interest in an image")
    parser.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()
    process(args.image_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

