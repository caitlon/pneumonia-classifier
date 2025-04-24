"""Script for running the API server."""

import argparse
import os

import uvicorn

from pneumonia_classifier.config import API_CONFIG


def main() -> None:
    """Starts the API server."""
    parser = argparse.ArgumentParser(description="Run API for pneumonia classification")
    parser.add_argument(
        "--host", type=str, default=API_CONFIG["host"], help="Host to run on"
    )
    parser.add_argument(
        "--port", type=int, default=API_CONFIG["port"], help="Port to run on"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model (if not specified, default path is used)",
    )
    parser.add_argument("--reload", action="store_true", help="Reload on file changes")

    args = parser.parse_args()

    # Set environment variable with model path
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    # Run application with uvicorn
    uvicorn.run(
        "pneumonia_classifier.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
