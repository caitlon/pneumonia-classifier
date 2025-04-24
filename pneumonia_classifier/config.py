"""Configuration parameters for the pneumonia classifier."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Base directory of the project
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directory paths
DATA_DIR = os.path.join(BASE_DIR, "data")
CHEST_XRAY_DIR = os.path.join(DATA_DIR, "chest_xray")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_classifier.pth")

# MLflow configuration
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
EXPERIMENT_NAME = "pneumonia_classification"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# Model configuration
INPUT_SIZE = 224  # Size of the input images after resizing
BATCH_SIZE = 32   # Batch size for training and evaluation
NUM_WORKERS = 4   # Number of workers for data loading

# Class names
CLASSES = ["NORMAL", "PNEUMONIA"]

# Normalization parameters (ImageNet statistics)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Training configuration
TRAIN_CONFIG = {
    "epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
                    If None, default hyperparameters are used.
    
    Returns:
        Dictionary with configuration parameters
    """
    if config_path is None:
        return TRAIN_CONFIG.copy()
    
    # Load configuration from file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override default parameters with values from file
    result = TRAIN_CONFIG.copy()
    result.update(config)
    
    return result

# Model settings
MODEL_CONFIG: Dict[str, Any] = {
    "model_name": "resnet18",
    "pretrained": True,
    "num_classes": 2,
    "class_names": CLASSES,
}

# Training settings
TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "learning_rate": TRAIN_CONFIG["learning_rate"],
    "epochs": TRAIN_CONFIG["epochs"],
    "default_model_path": MODEL_PATH,
    "early_stopping_patience": TRAIN_CONFIG["early_stopping_patience"],
}

# Image transformation settings
IMAGE_SIZE = (INPUT_SIZE, INPUT_SIZE)

# API settings
API_CONFIG: Dict[str, Any] = {
    "title": "Pneumonia Classification API",
    "description": "API for classifying pneumonia in chest X-ray images",
    "version": "1.0.0",
    "host": API_HOST,
    "port": API_PORT,
}

# MLflow settings
MLFLOW_CONFIG: Dict[str, Any] = {
    "experiment_name": EXPERIMENT_NAME,
    "tracking_uri": MLFLOW_TRACKING_URI,
    "artifacts_dir": os.path.join(MLFLOW_TRACKING_URI, "artifacts"),
    "registry_uri": f"sqlite:///{os.path.join(MLFLOW_TRACKING_URI, 'mlflow.db')}",
    "ui_port": 5000,
} 