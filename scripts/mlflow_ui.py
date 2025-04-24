"""Script for running MLflow UI."""

import argparse
import os

import mlflow

from pneumonia_classifier.config import MLFLOW_CONFIG


def main():
    """Starts MLflow UI for experiment visualization."""
    parser = argparse.ArgumentParser(description="Run MLflow UI")
    parser.add_argument(
        "--port", type=int, default=MLFLOW_CONFIG["ui_port"], help="Port to run UI on"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to run UI on"
    )

    args = parser.parse_args()

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])

    # Display startup information
    print(f"Starting MLflow UI at http://{args.host}:{args.port}")
    print(f"Tracking URI: {MLFLOW_CONFIG['tracking_uri']}")
    print(f"Experiment name: {MLFLOW_CONFIG['experiment_name']}")
    print("Press Ctrl+C to stop")

    # Start UI
    os.system(f"mlflow ui --host {args.host} --port {args.port}")


if __name__ == "__main__":
    main()
