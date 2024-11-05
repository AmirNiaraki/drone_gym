import cv2

# Load the image
image_path = "./2021-7-13-padded.png"  # Replace with your image file path
image = cv2.imread(image_path)

# Check if the image was loaded
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Resize the image to make it smaller (e.g., 50% of the original size)
    scale_percent = 10  # Percentage of the original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))

    # Define the initial position of the square
    square_top_left = [50, 50]
    square_size = 10  # Size of the square

    # Function to draw the square on the image
    def draw_square(image, top_left):
        bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
        color = (0, 255, 0)  # Green color in BGR format
        thickness = 2
        cv2.rectangle(image, tuple(top_left), bottom_right, color, thickness)

    # Main loop to display the image and move the square
    while True:
        # Create a copy of the resized image to draw on
        display_image = resized_image.copy()
        draw_square(display_image, square_top_left)

        # Display the image with the square
        cv2.imshow("Image with Movable Square", display_image)

        # Wait for a key press
        key = cv2.waitKey(0)

        # Move the square based on the key pressed
        if key == 27:  # ESC key to exit
            break
        elif key == ord("q"):  # Press 'q' to quit
            break
        elif key == ord("h"):  # Left arrow key
            square_top_left[0] -= 10
        elif key == ord("l"):  # Right arrow key
            square_top_left[0] += 10
        elif key == ord("k"):  # Up arrow key
            square_top_left[1] -= 10
        elif key == ord("j"):  # Down arrow key
            square_top_left[1] += 10
        elif key == ord(" "):  # Down arrow key
            square_size += 10

        # Prevent the square from going out of bounds
        square_top_left[0] = max(0, min(square_top_left[0], width - square_size))
        square_top_left[1] = max(0, min(square_top_left[1], height - square_size))

        # Calculate the scaling factor
        scale_x = image.shape[1] / width
        scale_y = image.shape[0] / height

        # Map the coordinates of the square to the original image
        orig_top_left_x = int(square_top_left[0] * scale_x)
        orig_top_left_y = int(square_top_left[1] * scale_y)
        orig_bottom_right_x = int((square_top_left[0] + square_size) * scale_x)
        orig_bottom_right_y = int((square_top_left[1] + square_size) * scale_y)

        # Crop the original image
        cropped_image = image[orig_top_left_y:orig_bottom_right_y, orig_top_left_x:orig_bottom_right_x]

        # Save or display the cropped image
        cv2.imshow("Cropped Image", cropped_image)
    cv2.destroyAllWindows()
