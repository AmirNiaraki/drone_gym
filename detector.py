import sys
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

sys.path.append("pytorch-retinanet")
from retinanet.dataloader import Normalizer, Resizer


class BaseDetector(ABC):
    @abstractmethod
    def preprocess(self, image):
        """Preprocess image before inference"""

    @abstractmethod
    def predict(self, image):
        """Run inference on preprocessed image"""

    @abstractmethod
    def postprocess(self, predictions):
        """Postprocess model outputs"""

    @abstractmethod
    def infer(self, image):
        """Perform all the steps of inference"""


from model_loader import load_retinanet


class RetinaNetDetector(BaseDetector):
    def __init__(self, model_config, confidence_threshold=0.5):
        self.model = load_retinanet(model_config)

        self.confidence_threshold = confidence_threshold
        self.normalizer = Normalizer()
        self.resizer = Resizer()
        self.device = next(self.model.parameters()).device

    def preprocess(self, image):
        # NOTE: this is so that the model have the same type of data it was
        # trained in
        image = image.astype(np.float32) / 255.0
        sample = {"img": image, "annot": np.zeros((0, 5))}
        sample = self.normalizer(sample)
        sample = self.resizer(sample)
        return (
            sample["img"].unsqueeze(0).permute(0, 3, 1, 2).to(self.device),
            sample["scale"],
        )

    def predict(self, preprocessed_data):
        input_tensor, scale = preprocessed_data
        with torch.no_grad():
            scores, _, boxes = self.model(input_tensor)
        return scores, boxes, scale

    def postprocess(self, predictions):
        scores, boxes, scale = predictions
        scores = scores.cpu().numpy()
        boxes = boxes.cpu().numpy()

        # Adjust boxes back to original image scale
        boxes /= scale

        # Filter by confidence threshold
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        return boxes, scores

    def infer(self, image):
        input_tensor, scale = self.preprocess(image)
        predictions = self.predict((input_tensor.to(self.device), scale))
        return self.postprocess(predictions)


if __name__ == "__main__":
    # quick example of how to use the inferer
    from model_loader import ModelConfig

    # Load your model
    config = ModelConfig()
    model = RetinaNetDetector(config)

    # Load an image
    image = cv2.imread(
        "/Volumes/EX_DRIVE/new_git/images/NDVI/2021-7-13_field1_w0_h0.png"
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # NOTE: this can be included in the preprocessing pipeline
    # it all depends how the image is loaded.
    # image = image.astype(np.float32) / 255.0

    # Run inference
    boxes, scores = model.infer(image)
    for box in boxes:
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
