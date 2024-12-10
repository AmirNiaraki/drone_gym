import pickle
import sys
from abc import ABC, abstractmethod
import image_resizer

import cv2
import numpy as np
import torch
from sklearn import cluster
from sklearn.cluster import DBSCAN

sys.path.append("pytorch-retinanet")
from retinanet.dataloader import Normalizer, Resizer

from model_loader import ModelConfig
import logging


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
        logging.info(f"Detected {len(boxes)} objects at : {boxes}")

        return boxes, scores

    def infer(self, image):
        input_tensor, scale = self.preprocess(image)
        predictions = self.predict((input_tensor.to(self.device), scale))
        return self.postprocess(predictions)


class ClusteringDetector(BaseDetector):
    def __init__(self, model_path, selected_label=1):
        self.kmeans_cluster = self.load_kmeans(model_path)
        # for repeatability
        self.cluster_centers = self.kmeans_cluster.cluster_centers_
        logging.info(f"Cluster centers: {self.cluster_centers}")
        sorted_indices = np.argsort(self.cluster_centers[:, 0])  # Or sort by another dimension
        self.cluster_centers = self.cluster_centers[sorted_indices]

        self.selected_label = selected_label

    @staticmethod
    def fit_kmeans(image, n_clusters=4):
        image = ClusteringDetector.preprocess(image)
        x, y, z = image.shape
        image_2d = image.reshape(x * y, z)
        kmeans_cluster = cluster.KMeans(n_clusters=n_clusters)
        kmeans_cluster.fit(image_2d)
        ClusteringDetector.save_kmeans(kmeans_cluster)

    @staticmethod
    def save_kmeans(kmeans_cluster, path="./weights/kmeans_model.pkl"):
        # Save the model
        with open(path, "wb") as f:
            pickle.dump(kmeans_cluster, f)
        logging.info("done saving")

    def load_kmeans(self, path):
        with open(f"{path}", "rb") as f:
            kmeans_cluster = pickle.load(f)

        return kmeans_cluster

    @staticmethod
    def preprocess(image):
        return image / 255

    def segment_image(self, image, anomaly_label=3, save=True):
        x, y, z = image.shape
        img_2d = image.reshape(x * y, z)
        cluster_labels = self.kmeans_cluster.predict(img_2d)

        segmented_image = self.kmeans_cluster.cluster_centers_[cluster_labels].reshape(
            x, y, z
        )

        # Select a specific center and make it black
        segmented_image[cluster_labels.reshape(x, y) == anomaly_label] = [0, 0, 0]

        if save:
            cv2.imwrite("images/segmented_image.png", segmented_image * 255)

        return segmented_image

    def kmeans_predict(self, image):
        # used for clustering based on color
        x, y, z = image.shape
        img_2d = image.reshape(x * y, z)
        cluster_labels = self.kmeans_cluster.predict(img_2d)

        label_regions = np.where(cluster_labels.reshape(x, y) == self.selected_label)
        label_regions = np.asarray(label_regions).T
        return label_regions

    def predict(self, image):
        # used for clustering based on color
        label_regions = self.kmeans_predict(image)

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
        logging.info(f"Detected {len(predictions)} objects at : {predictions}")
        return predictions

    def infer(self, image):
        image = self.preprocess(image).copy()
        prediction = self.predict(image)
        return self.postprocess(prediction)


if __name__ == "__main__":
    # NOTE: This is used to train the kmeans model
    # image = cv2.imread("images/resize.png")
    # ClusteringDetector.fit_kmeans(image, n_clusters=4)

    model = ClusteringDetector("weights/kmeans_model.pkl", selected_label=2)
    # Load your model
    # config = ModelConfig()
    # model = RetinaNetDetector(config, confidence_threshold=0.8)

    for i in range(1, 13):
        img_path = f"images/field{i}/field{i}.png"
        # Load an image
        # img_path = "images/field1/field1.png"
        image = cv2.imread(img_path)
        image = image_resizer.resize_image(image, max_height=5000)

        # for visualization
        def draw_boxes(image, boxes, color=(0, 255, 0)):
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            return image

        image = model.preprocess(image)
        segmented_image = model.segment_image(image, save=False)

        SEG_PATH = img_path.replace(".png", "_segmented.png")
        cv2.imwrite(SEG_PATH, segmented_image * 255)
        print('done with', SEG_PATH)

    # segmented_image = model.cluster_centers[labels]
    # print(segmented_image.shape)

    # Run inference
    # boxes, _ = model.infer(image.copy())
    # result_image = draw_boxes(image, boxes)
    # cv2.imshow("image", segment_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
