"""
ResNet model implementation for pneumonia classification from chest X-ray images.
"""
from typing import Dict, Optional, Union, cast

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from pneumonia_classifier.config import (
    IMAGE_SIZE,
    MODEL_CONFIG,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

# Правильный регистр имен классов для API
API_CLASS_NAMES = ["Normal", "Pneumonia"]


def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Create a ResNet model for pneumonia classification.
    
    Args:
        num_classes: Number of output classes (default: 2 - normal and pneumonia)
        pretrained: Whether to use a pretrained model (default: True)
        
    Returns:
        model: The ResNet model
    """
    # Load a pre-trained ResNet-18 model
    model = models.resnet18(pretrained=pretrained)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the last fully connected layer
    fc_layer = cast(nn.Linear, model.fc)
    in_features = fc_layer.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return cast(nn.Module, model)


def load_model(model_path: str, device: Union[str, torch.device] = "cpu") -> nn.Module:
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model to ('cpu' or 'cuda')
        
    Returns:
        model: The loaded model
    """
    # Create a new model with the same architecture
    model = create_model(num_classes=MODEL_CONFIG["num_classes"], pretrained=False)
    
    # Load model state dict from the checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return cast(nn.Module, model)


def predict(image_input: Union[str, Image.Image, torch.Tensor], model: Optional[nn.Module] = None, model_path: Optional[str] = None, device: str = "cpu") -> Dict[str, Union[str, int, float]]:
    """
    Make a prediction for an image.
    
    Args:
        image_input: Path to the image file, PIL Image object or preprocessed image tensor
        model: Pre-loaded model instance (if provided, model_path is ignored)
        model_path: Path to the model checkpoint file (optional if model is provided)
        device: Device to use for inference ('cpu' or 'cuda')
        
    Returns:
        prediction: Dictionary with class name, id and probability
    """
    # Check if input is already a tensor (for testing)
    if isinstance(image_input, torch.Tensor):
        image_tensor = image_input.to(device)
    else:
        # Load the image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input
        
        # Transform the image for the model
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Use provided model or load from path
    if model is not None:
        model_to_use = model
    else:
        # Load the model if path provided or use default
        if model_path is None:
            # Use default model path from config
            from pneumonia_classifier.config import TRAINING_CONFIG
            model_path = TRAINING_CONFIG["default_model_path"]
        
        model_to_use = load_model(model_path, device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model_to_use(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get the most likely class
    max_prob, pred_idx = torch.max(probabilities, 0)
    index = pred_idx.item()
    
    # Используем имена классов с правильным регистром для API
    class_name = API_CLASS_NAMES[index]
    
    return {
        "class_name": class_name,
        "class_id": index,
        "probability": max_prob.item()
    } 