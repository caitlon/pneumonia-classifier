"""Integration tests for pneumonia classification system."""

import io
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from unittest import mock

from pneumonia_classifier.api.main import app
from pneumonia_classifier.models.resnet import create_model, predict


@pytest.fixture
def client() -> TestClient:
    """Test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_model(monkeypatch: pytest.MonkeyPatch, cpu_device: torch.device) -> None:
    """Mock model for testing."""
    # Create model - using CPU
    test_model = create_model()
    test_model = test_model.to(cpu_device)

    # Mock predictions to return higher probability for Pneumonia (class 1)
    def mock_forward(*args: object, **kwargs: object) -> torch.Tensor:
        # Return artificial logits (prediction for class 1 - Pneumonia with high probability)
        return torch.tensor([[1.0, 5.0]])  # Significantly higher for Pneumonia

    # Replace forward method
    test_model.forward = mock_forward  # type: ignore

    # Replace global model
    monkeypatch.setattr("pneumonia_classifier.api.main.model", test_model)
    
    monkeypatch.setattr("pneumonia_classifier.utils.get_device", lambda: cpu_device)


def test_model_to_api_integration(cpu_device: torch.device) -> None:
    """Test the integration between model and API."""
    # First, create and check the model
    model = create_model()
    assert isinstance(model, torch.nn.Module)
    
    # Create test image and convert it to tensor
    img = Image.new("RGB", (224, 224), color="white")
    img_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    
    # Get prediction directly from model
    model_result = predict(img_tensor, model=model, device=str(cpu_device))
    
    # Check that prediction has expected structure
    assert "class_name" in model_result
    assert "class_id" in model_result
    assert "probability" in model_result


def test_end_to_end(client: TestClient, get_test_image_bytes: bytes, mock_model: None) -> None:
    """Test end-to-end flow from image to prediction."""
    # Use test image
    img_data = io.BytesIO(get_test_image_bytes)
    
    # Send request to API
    response = client.post(
        "/predict", files={"file": ("test.jpg", img_data, "image/jpeg")}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "class_name" in data
    assert "class_id" in data
    assert "probability" in data
    assert "filename" in data
    assert "content_type" in data
    
    # Check that class matches mock (should be Pneumonia)
    assert data["class_name"] == "Pneumonia"
    assert data["class_id"] == 1
    assert 0.8 <= data["probability"] <= 1.0


@pytest.mark.parametrize("image_size", [(224, 224), (512, 512), (100, 100)])
def test_api_different_image_sizes(
    client: TestClient, 
    image_size: tuple[int, int],
    mock_model: None
) -> None:
    """Test API with different image sizes."""
    # Create image of specified size
    width, height = image_size
    img = Image.new("RGB", (width, height), color="white")
    
    # Save image to byte stream
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    # Send request to API
    response = client.post(
        "/predict", 
        files={"file": (f"test_{width}x{height}.jpg", img_byte_arr, "image/jpeg")}
    )
    
    # Check response - API should handle any image size
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "class_name" in data
    assert "class_id" in data
    assert "probability" in data
    
    # Check that class matches mock (should be Pneumonia)
    assert data["class_name"] == "Pneumonia"
    assert data["class_id"] == 1
    assert 0.8 <= data["probability"] <= 1.0 