"""
This script is the inference engine containing multiple objects (each for one model)
that receive a frame and return bounding box of the detected objects within the frames.
There are three models:
1- low_fidelity: it expects a gray scale image and counts the black pixels in the frame
2- DCNN: is a double clustering method that bins the pixels of a certain color and then
         clusters the bins of the same class that are next to each other and returns bb
3- retinaNet: is a deep learning model that returns bounding boxes of the detected objects
"""

import logging
import sys

import cv2
import numpy as np
import torch

sys.path.append("pytorch-retinanet")

from retinanet.dataloader import Normalizer, Resizer


class Inferer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = "low_fidelity"

        logging.info(f"Using model: {self.model}")

    def infer(self, frame):
        # TODO: remove the battery bar from frame entirely in the project
        # count the number of black pixels within the frame
        if self.model == "low_fidelity":
            cv2.imwrite("images/frame_for_inference.jpg", frame)
            score = self.count_black_pixels(frame)

    def count_black_pixels(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_pixels = gray.size - cv2.countNonZero(gray)
        logging.info(f"Number of black pixels: {black_pixels}")
        return black_pixels

    def write_sample_to_disk(self, frame):
        cv2.imwrite("images/frame_for_inference.jpg", frame)


class InferencePreprocessor:
    def __init__(self):
        self.normalizer = Normalizer()
        self.resizer = Resizer()

    def preprocess_image(self, image):
        """
        Preprocess a single image for inference
        Args:
            image: numpy array of shape (H, W, C) in RGB format
        Returns:
            tensor: preprocessed image tensor ready for model input
            float: scale factor used in resizing
        """
        # Create sample dict expected by transforms
        sample = {"img": image, "annot": np.zeros((0, 5))}

        # Apply normalizer and resizer
        sample = self.normalizer(sample)
        sample = self.resizer(sample)

        # Get the processed image and scale
        return sample["img"].unsqueeze(0).permute(0, 3, 1, 2), sample["scale"]


def run_inference(model, image, confidence_threshold=0.5):
    """
    Run inference on a single image
    Args:
        model: loaded RetinaNet model
        image: numpy array of shape (H, W, C) in RGB format
        confidence_threshold: float, minimum confidence score for detections
    Returns:
        boxes: numpy array of bounding boxes
        scores: numpy array of confidence scores
    """
    preprocessor = InferencePreprocessor()

    # Preprocess image
    input_tensor, scale = preprocessor.preprocess_image(image)

    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        scores, _, boxes = model(input_tensor)

        # Filter by confidence
        scores = scores.cpu().numpy()
        boxes = boxes.cpu().numpy()

        # Adjust boxes back to original image scale
        boxes /= scale

        # Filter by confidence threshold
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]

    return boxes, scores



if __name__ == "__main__":
    # quick example of how to use the inferer
    from model_loader import ModelConfig, load_retinanet
    # Load your model
    config = ModelConfig()
    model = load_retinanet(config)

    # Load an image
    image = cv2.imread(
        "/Volumes/EX_DRIVE/new_git/images/NDVI/2021-7-13_field1_w0_h0.png"
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # NOTE: this can be included in the preprocessing pipeline
    # it all depends how the image is loaded.
    image = image.astype(np.float32) / 255.0

    # Run inference
    boxes, scores = run_inference(model, image, confidence_threshold=0.5)
    for box in boxes:
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
