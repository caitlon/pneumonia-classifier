"""Tests for the pneumonia classification model."""

import pytest
import torch
import torch.nn as nn

from pneumonia_classifier.models.resnet import create_model, load_model, predict
from pneumonia_classifier.utils import save_model


def test_create_model():
    """Test model creation."""
    model = create_model()
    
    # Check model type
    assert isinstance(model, nn.Module)
    
    # Check that the last layer has 2 outputs
    assert model.fc.out_features == 2
    
    # Check that the model handles input shape correctly
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    assert output.shape == (1, 2)


def test_save_and_load_model(tmp_path):
    """Test saving and loading the model."""
    # Create model
    model = create_model()
    
    # Save model
    model_path = tmp_path / "test_model.pth"
    save_model(model.state_dict(), model_path)
    
    # Check that the file exists
    assert model_path.exists()
    
    # Load model
    loaded_model = load_model(model_path)
    
    # Check model type
    assert isinstance(loaded_model, nn.Module)
    
    # Check that the last layer has 2 outputs
    assert loaded_model.fc.out_features == 2
    
    # Check output shape
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = loaded_model(test_input)
    assert output.shape == (1, 2)


def test_predict():
    """Test prediction function."""
    # Create model
    model = create_model()
    
    # Create test input
    test_input = torch.randn(1, 3, 224, 224)
    
    # Perform prediction
    result = predict(model, test_input)
    
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