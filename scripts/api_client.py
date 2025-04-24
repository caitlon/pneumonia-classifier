#!/usr/bin/env python
"""Simple client for interacting with the pneumonia classification API."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import requests


def predict_with_api(image_path: str, api_url: str) -> Optional[Dict[str, Any]]:
    """
    Send an image to the API for prediction.

    Args:
        image_path: Path to the image file
        api_url: URL of the API endpoint

    Returns:
        API response as a dictionary or None if error occurred
    """
    # Ensure the API URL has the correct endpoint
    if not api_url.endswith("/predict"):
        api_url = f"{api_url.rstrip('/')}/predict"

    try:
        # Prepare the image file
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}

            # Make the request
            print(f"Sending request to {api_url}...")
            response = requests.post(api_url, files=files, timeout=10)

        # Check if the request was successful
        if response.status_code == 200:
            result: Dict[str, Any] = response.json()
            print("\nPrediction Results:")
            print(f"Diagnosis: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']}%")
            print("\nDetailed Probabilities:")
            for class_name, prob in result["probabilities"].items():
                print(f"  {class_name}: {prob}%")
            return result
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        print("Please check that the API server is running and accessible.")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Connection timeout: {e}")
        print("The server is taking too long to respond.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Client for pneumonia classification API"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument("--api_url", type=str, required=True, help="URL of the API")

    args = parser.parse_args()

    # Check if the image file exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file {args.image_path} does not exist")
        return

    # Make the prediction
    result = predict_with_api(args.image_path, args.api_url)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
