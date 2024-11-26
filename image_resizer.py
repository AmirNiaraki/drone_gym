import cv2
import os
import numpy as np

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    return img

def resize_image(img, max_height=5000):
    height = img.shape[0]
    ratio = max_height / height
    dim = (int(img.shape[1] * ratio), max_height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def save_image(img, original_path, suffix='_resized', extension='.png'):
    img_name, _ = os.path.splitext(original_path)
    new_img_path = f"{img_name}{suffix}{extension}"
    cv2.imwrite(new_img_path, img)
    return new_img_path

def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_red_mask(img, red_value=100, tolerance=50):
    lower_bound = np.array([0, 0, red_value - tolerance])
    upper_bound = np.array([255, 255, red_value + tolerance])
    mask = cv2.inRange(img, lower_bound, upper_bound)
    img[mask > 0] = [0, 0, 255]  # Paint those pixels completely red
    return img

def process_image(img_path, apply_mask=False):
    img = load_image(img_path)
    print(f"Original image shape: {img.shape}")
    resized_img = resize_image(img)
    print(f"Resized image shape: {resized_img.shape}")
    show_image(resized_img)
    new_img_path = save_image(resized_img, img_path)
    print(f"Resized image saved at: {new_img_path}")

    if apply_mask:
        red_masked_img = create_red_mask(resized_img)
        red_masked_img_path = save_image(red_masked_img, img_path, suffix='_red_masked')
        print(f"Red masked image saved at: {red_masked_img_path}")

def main(img_path, apply_mask=False):
    process_image(img_path, apply_mask)

if __name__ == "__main__":
    img_path = 'images/field7_NDVI_resized.png'
    # img_path = 'images/field7_RGB_resized.png'
    main(img_path, apply_mask=True)
