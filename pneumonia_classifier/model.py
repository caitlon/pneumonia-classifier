"""Model definition for pneumonia classification."""

from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
from torchvision import models


def create_model() -> nn.Module:
    """
    Creates a model based on pre-trained ResNet18.
    
    Returns:
        model: The pneumonia classification model
    """
    # Load pre-trained model
    model = models.resnet18(pretrained=True)
    
    # Freeze base model weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the last layer with a new one with 2 outputs (Normal/Pneumonia)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Unfreeze the last few layers for fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
    
    return model


def save_model(model_state: Dict[str, Any], save_path: str = "model.pth") -> None:
    """
    Saves the model to a file.
    
    Args:
        model_state: Model state to save
        save_path: Path to save the model
    """
    torch.save(model_state, save_path)


def load_model(model_path: str, device: Optional[torch.device] = None) -> nn.Module:
    """
    Loads a trained model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model to
        
    Returns:
        The loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with correct architecture
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load weights from file
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model 