"""Tests for the pneumonia classification model."""

from pathlib import Path
import io

import torch
import torch.nn as nn
import pytest
from PIL import Image

from pneumonia_classifier.models.resnet import create_model, load_model, predict
from pneumonia_classifier.utils import save_model


def test_create_model() -> None:
    """Test model creation."""
    model = create_model()

    # Check model type
    assert isinstance(model, nn.Module)

    # Check that the last layer has 2 outputs
    assert model.fc.out_features == 2  # type: ignore

    # Check that the model handles input shape correctly
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    assert output.shape == (1, 2)


def test_save_and_load_model(tmp_path: Path, cpu_device: torch.device) -> None:
    """Test saving and loading the model."""
    # Create model
    model = create_model()

    # Save model
    model_path = tmp_path / "test_model.pth"
    save_model(model.state_dict(), str(model_path))

    # Check that the file exists
    assert model_path.exists()

    # Load model - using device fixture
    loaded_model = load_model(str(model_path), device=str(cpu_device))

    # Check model type
    assert isinstance(loaded_model, nn.Module)

    # Check that the last layer has 2 outputs
    assert loaded_model.fc.out_features == 2  # type: ignore

    # Check output shape
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = loaded_model(test_input)
    assert output.shape == (1, 2)


def test_predict(cpu_device: torch.device) -> None:
    """Test prediction function."""
    # Create model
    model = create_model()

    # Create test input
    test_input = torch.randn(1, 3, 224, 224)

    # Perform prediction - using device fixture
    result = predict(test_input, model=model, device=str(cpu_device))

    # Check result structure
    assert "class_name" in result
    assert "class_id" in result
    assert "probability" in result

    # Check result types
    assert isinstance(result["class_name"], str)
    assert isinstance(result["class_id"], int)
    assert isinstance(result["probability"], float)

    # Check value ranges
    assert 0 <= result["class_id"] <= 1
    assert 0 <= result["probability"] <= 1


# Parameterized test for testing different image sizes
@pytest.mark.parametrize("image_size", [(224, 224), (256, 256), (196, 196)])
def test_model_different_image_sizes(image_size: tuple[int, int], cpu_device: torch.device) -> None:
    """Test model with different image sizes."""
    model = create_model()
    
    # Create test image of specified size
    width, height = image_size
    img = Image.new("RGB", (width, height), color="white")
    
    # Convert to tensor
    img_tensor = torch.zeros((1, 3, height, width), dtype=torch.float32)
    
    # Get prediction
    result = predict(img_tensor, model=model, device=str(cpu_device))
    
    # Check result structure
    assert "class_name" in result
    assert "class_id" in result
    assert "probability" in result
