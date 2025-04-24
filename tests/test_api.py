"""Tests for the pneumonia classification API."""

import io
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from pneumonia_classifier.api.main import app
from pneumonia_classifier.models.resnet import create_model


@pytest.fixture
def client() -> TestClient:
    """Test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_model(monkeypatch: pytest.MonkeyPatch, cpu_device: torch.device) -> Any:
    """Mock model for testing."""
    # Create model - using CPU
    test_model = create_model()
    test_model = test_model.to(cpu_device)

    # Mock predictions to return higher probability for Pneumonia (class 1)
    def mock_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        # Return artificial logits (prediction for class 1 - Pneumonia with high probability)
        return torch.tensor([[1.0, 5.0]])  # Significantly higher for Pneumonia

    # Replace forward method
    test_model.forward = mock_forward

    # Replace global model
    monkeypatch.setattr("pneumonia_classifier.api.main.model", test_model)
    
    monkeypatch.setattr("pneumonia_classifier.utils.get_device", lambda: cpu_device)

    return test_model


def test_root_endpoint(client: TestClient) -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "usage" in data


def test_health_endpoint(client: TestClient, mock_model: Any) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint(client: TestClient, mock_model: Any, get_test_image_bytes: bytes) -> None:
    """Test prediction endpoint."""
    # Use fixture to get test image
    img_byte_arr = io.BytesIO(get_test_image_bytes)

    # Send request
    response = client.post(
        "/predict", files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
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

    # Check response content
    assert data["class_name"] == "Pneumonia"  # Should match the mock
    assert data["class_id"] == 1
    assert 0.8 <= data["probability"] <= 1.0
    assert data["filename"] == "test.jpg"
    assert data["content_type"] == "image/jpeg"


def test_predict_wrong_file_type(client: TestClient, mock_model: Any) -> None:
    """Test handling of incorrect file type."""
    # Create text file
    text_file = io.BytesIO(b"This is not an image")

    # Send request
    response = client.post(
        "/predict", files={"file": ("test.txt", text_file, "text/plain")}
    )

    # Check response
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "image" in data["detail"]


# Parameterized test for different image formats
@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "BMP"])
def test_predict_different_image_formats(
    client: TestClient, 
    mock_model: Any, 
    image_format: str
) -> None:
    """Test prediction with different image formats."""
    # Create image of the specified format
    img = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=image_format)
    img_byte_arr.seek(0)
    
    # Define MIME type
    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "BMP": "image/bmp"
    }
    
    # Send request
    response = client.post(
        "/predict", 
        files={"file": (f"test.{image_format.lower()}", img_byte_arr, mime_types[image_format])}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "class_name" in data
    assert "class_id" in data
    assert "probability" in data
    
    # Check response content
    assert data["class_name"] == "Pneumonia"  # Should match the mock
    assert data["class_id"] == 1
    assert data["content_type"] == mime_types[image_format]
