# handler.py
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Any
from PIL import Image
import base64
import torch
import io


class EndpointHandler:
    def __init__(self, path=""):
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        data args:
            inputs (:obj: `str`): Base64 encoded image
            candidate_labels (:obj: `List[str]`): List of candidate labels for classification
        Return:
            A :obj:`list` of :obj:`dict`: Predictions will be serialized and returned
        """
        # Get inputs
        image_base64 = data.pop("inputs", None)
        candidate_labels = data.pop("candidate_labels", [])

        if not image_base64 or not candidate_labels:
            return [{"error": "Missing inputs or candidate_labels"}]

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Prepare inputs
        inputs = self.processor(
            text=candidate_labels, images=image, return_tensors="pt", padding=True
        )

        # Generate predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Format predictions
        predictions = []
        for i, label in enumerate(candidate_labels):
            predictions.append({"label": label, "score": probs[0][i].item()})

        # Sort predictions by score in descending order
        predictions.sort(key=lambda x: x["score"], reverse=True)

        return predictions
