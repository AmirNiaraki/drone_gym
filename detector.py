import pickle
import sys
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from sklearn import cluster
from sklearn.cluster import DBSCAN

sys.path.append("pytorch-retinanet")
from retinanet.dataloader import Normalizer, Resizer

from model_loader import ModelConfig


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
    def __init__(self, model_config: ModelConfig, confidence_threshold: float = 0.5):
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


class ClusteringDetector(BaseDetector):
    def __init__(self, model_path, selected_label=1):
        self.kmeans_cluster = self.load_kmeans(model_path)
        self.cluster_centers = self.kmeans_cluster.cluster_centers_
        self.selected_label = selected_label

    def fit_kmeans(self, image):
        x, y, z = image.shape
        image_2d = image.reshape(x * y, z)
        image_2d.shape
        kmeans_cluster = cluster.KMeans(n_clusters=4)
        kmeans_cluster.fit(image_2d)
        self.save_kmeans(kmeans_cluster)

    def save_kmeans(self, kmeans_cluster):
        # Save the model
        with open("kmeans_model.pkl", "wb") as f:
            pickle.dump(kmeans_cluster, f)
        print("done saving")

    def load_kmeans(self, path):
        with open(f"{path}", "rb") as f:
            kmeans_cluster = pickle.load(f)

        return kmeans_cluster

    def preprocess(self, image):
        return image / 255

    def predict(self, image):
        # used for clustering based on color
        x, y, z = image.shape
        img_2d = image.reshape(x * y, z)
        cluster_labels = self.kmeans_cluster.predict(img_2d)

        label_regions = np.where(cluster_labels.reshape(x, y) == self.selected_label)
        label_regions = np.asarray(label_regions).T

        # used for the clustering based on localization
        boxes = []
        if label_regions.shape[0] != 0:
            clustering = DBSCAN(eps=10, min_samples=300).fit(label_regions)

            sel = 0
            segmented_image2 = image.copy()
            for sel in np.unique(clustering.labels_):
                ind = clustering.labels_ == sel
                y1, y2 = np.min(label_regions[ind, 0]), np.max(label_regions[ind, 0])
                x1, x2 = np.min(label_regions[ind, 1]), np.max(label_regions[ind, 1])
                if sel != -1:
                    boxes.append([x1, y1, x2, y2])
                    segmented_image2 = cv2.rectangle(
                        segmented_image2, (x1, y1), (x2, y2), (255, 255, 255), 10
                    )
        boxes = np.array(boxes)

        return boxes, None

    def postprocess(self, predictions):
        # Convert clustering results to bounding box format
        # This would depend on your specific clustering approach
        return predictions

    def infer(self, image):
        image = self.preprocess(image).copy()
        prediction = self.predict(image)
        return self.postprocess(prediction)


if __name__ == "__main__":
    # NOTE: This is used to train the kmeans model
    # image = cv2.imread("/Volumes/EX_DRIVE/new_git/images/resize.png") / 255
    model = ClusteringDetector("kmeans_model.pkl")
    # Load your model
    # config = ModelConfig()
    # model = RetinaNetDetector(config)

    # Load an image
    image = cv2.imread(
        "/Volumes/EX_DRIVE/new_git/images/NDVI/2021-7-13_field1_w0_h0.png"
    )
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