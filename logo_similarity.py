import re
import cv2
import numpy as np
from PIL import Image
from google.cloud import vision
from typing import Tuple, Optional
from sklearn.cluster import KMeans
from goo_secrets import fetch_local_credentials

try:
    vs_client = vision.ImageAnnotatorClient()
except Exception as _e:
    creds = fetch_local_credentials()
    vs_client = vision.ImageAnnotatorClient(credentials=creds)


class ImageSimilarity:
    def __init__(self):
        self.vision_client = vs_client
        self.feature_extractor = cv2.SIFT_create()

    def compare_images(self, img_a: Image.Image, img_b: Image.Image) -> float:
        """
        Compare two images and return a similarity score.

        :param img_a: First image to compare
        :param img_b: Second image to compare
        :return: Similarity score between 0 and 1
        """
        # Preprocess images
        img_a_processed = self.preprocess_image(img_a)
        img_b_processed = self.preprocess_image(img_b)

        # Compute similarities
        color_similarity = self.compute_color_similarity(
            img_a_processed, img_b_processed
        )
        feature_similarity = self.compute_feature_similarity(
            img_a_processed, img_b_processed
        )
        text_similarity = self.compute_text_similarity(img_a_processed, img_b_processed)

        # Combine similarities
        similarities = [color_similarity, feature_similarity, text_similarity]
        valid_similarities = [s for s in similarities if s is not None]

        if not valid_similarities:
            return 0.0

        return sum(valid_similarities) / len(valid_similarities)

    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Preprocess the image for comparison."""
        img = img.convert("RGB")
        img = img.resize((224, 224))  # Resize for consistency
        return np.array(img)

    def compute_color_similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Compute color similarity between two images."""
        color_a = self.get_dominant_color(img_a)
        color_b = self.get_dominant_color(img_b)
        distance = np.linalg.norm(np.array(color_a) - np.array(color_b))
        max_distance = np.sqrt(255**2 * 3)  # Max possible distance in RGB space
        return 1 - (distance / max_distance)

    def get_dominant_color(self, img: np.ndarray) -> Tuple[int, int, int]:
        """Get the dominant color of an image."""
        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=1, n_init=10).fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]
        return tuple(map(int, dominant_color))

    def compute_feature_similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Compute feature similarity between two images using SIFT."""
        # Convert images to grayscale
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

        # Detect and compute SIFT features
        _, des_a = self.feature_extractor.detectAndCompute(gray_a, None)
        _, des_b = self.feature_extractor.detectAndCompute(gray_b, None)

        # If no features are found in either image, return 0 similarity
        if des_a is None or des_b is None:
            return 0.0

        # Use FLANN for fast feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des_a, des_b, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Compute similarity based on the number of good matches
        similarity = len(good_matches) / max(len(des_a), len(des_b))
        return min(similarity, 1.0)  # Ensure similarity is at most 1.0

    def compute_text_similarity(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> Optional[float]:
        """Compute text similarity between two images."""
        text_a = self.extract_text(img_a)
        text_b = self.extract_text(img_b)

        if text_a is None or text_b is None:
            return None

        if text_a == text_b:
            return 1.0

        # Compute Jaccard similarity for partial matches
        set_a = set(text_a.split())
        set_b = set(text_b.split())
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def extract_text(self, img: np.ndarray) -> Optional[str]:
        """Extract text from an image using Google Vision API."""
        img_bytes = self.numpy_to_bytes(img)
        image = vision.Image(content=img_bytes)
        response = self.vision_client.text_detection(
            image=image, image_context={"language_hints": ["en"]}
        )

        if response.error.message:
            return None

        texts = response.text_annotations
        if texts:
            return re.sub(r"[^\x00-\x7f]", r"", texts[0].description).upper()
        return ""

    @staticmethod
    def numpy_to_bytes(img: np.ndarray) -> bytes:
        """Convert numpy array to bytes."""
        is_success, buffer = cv2.imencode(".png", img)
        if not is_success:
            raise ValueError("Failed to convert image to bytes")
        return buffer.tobytes()


# Usage example:
# comparator = ImageSimilarity()
# similarity = comparator.compare_images(image1, image2)
# print(f"Similarity: {similarity:.2f}")

__all__ = ["ImageSimilarity"]
