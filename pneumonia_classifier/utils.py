"""
Utility functions for pneumonia classification model.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms

from pneumonia_classifier.config import INPUT_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

logger = logging.getLogger(__name__)


def process_image(image: Image.Image) -> torch.Tensor:
    """
    Process an image for model inference.

    Args:
        image: PIL Image to process

    Returns:
        Processed image tensor (1, C, H, W)
    """
    # Convert grayscale to RGB if needed
    if image.mode == "L":
        image = image.convert("RGB")

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    # Apply transformations
    img_tensor = transform(image)

    # Add batch dimension
    result: torch.Tensor = img_tensor.unsqueeze(0)
    return result


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration file
    """
    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Config saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        filepath: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    with open(filepath, "r") as f:
        config: Dict[str, Any] = json.load(f)
    logger.info(f"Config loaded from {filepath}")
    return config


def plot_metrics(
    metrics: Dict[str, Dict[str, List[float]]], save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation metrics.

    Args:
        metrics: Dictionary containing metrics history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train"]["loss"], label="Train Loss")
    plt.plot(metrics["val"]["loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train"]["acc"], label="Train Accuracy")
    plt.plot(metrics["val"]["acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Metrics plot saved to {save_path}")

    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close()


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate and save classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the report

    Returns:
        Classification report as dictionary
    """
    report_dict: Dict[str, Dict[str, Any]] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    if save_path:
        with open(save_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        logger.info(f"Classification report saved to {save_path}")

    return report_dict


def set_device() -> torch.device:
    """
    Set the device for training (CPU or GPU).

    Returns:
        torch.device object for the available device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def save_model(model_state: Dict[str, Any], save_path: str) -> None:
    """
    Saves the model to a file.

    Args:
        model_state: Model state to save
        save_path: Path where the model will be saved
    """
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model_state, save_path)
    print(f"Model saved to {save_path}")


def get_device() -> torch.device:
    """
    Determines the available device for computations.

    Returns:
        device: Computing device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Temporarily disabling MPS to avoid errors on Mac
    # elif torch.backends.mps.is_available():
    #     # Apple Silicon support
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Formats metrics for convenient display.

    Args:
        metrics: Dictionary with metrics

    Returns:
        String with formatted metrics
    """
    return ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
