"""Script for prediction using the trained model."""

import argparse
import os
from typing import Dict, Optional, Any

from PIL import Image

from pneumonia_classifier.config import TRAINING_CONFIG
from pneumonia_classifier.data.dataset import process_image
from pneumonia_classifier.models.resnet import load_model, predict
from pneumonia_classifier.utils import get_device


def predict_image_file(image_path: str, model_path: str) -> Optional[Dict[str, Any]]:
    """
    Makes a prediction based on an image file.

    Args:
        image_path: Path to the image
        model_path: Path to the model
        
    Returns:
        Dictionary with prediction results or None if error occurred
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Load the model
        device = get_device()
        model = load_model(model_path, device)

        # Process the image
        image_tensor = process_image(image).to(device)

        # Get prediction
        result = predict(image_tensor, model=model)
        
        prediction_result: Dict[str, Any] = {
            "class_name": result["class_name"],
            "class_id": result["class_id"],
            "probability": result["probability"]
        }

        # Display the result
        print(f"\nFile: {image_path}")
        print(f"Diagnosis: {prediction_result['class_name']}")
        print(
            f"Confidence: {prediction_result['probability']:.4f} ({prediction_result['probability']*100:.2f}%)"
        )

        return prediction_result

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def main() -> None:
    """Main function for prediction."""
    parser = argparse.ArgumentParser(
        description="Pneumonia prediction from X-ray image"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=TRAINING_CONFIG["default_model_path"],
        help="Path to the model",
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.image_path):
        print(f"Error: File {args.image_path} does not exist")
        return

    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return

    _ = predict_image_file(args.image_path, args.model_path)


if __name__ == "__main__":
    main()
