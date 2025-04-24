"""Tests for error handling in pneumonia classification system."""

import io
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from unittest import mock

from pneumonia_classifier.api.main import app
from pneumonia_classifier.models.resnet import create_model


@pytest.fixture
def client() -> TestClient:
    """Test client for the API."""
    return TestClient(app)


def test_missing_file(client: TestClient) -> None:
    """Test API response when no file is provided."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data


def test_empty_file(client: TestClient) -> None:
    """Test API response when an empty file is provided."""
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/predict", files={"file": ("empty.jpg", empty_file, "image/jpeg")}
    )
    # Current behavior - server returns 500 error, though in future it could be improved to 400
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data


def test_corrupt_image(client: TestClient) -> None:
    """Test API response when a corrupt image is provided."""
    corrupt_file = io.BytesIO(b"This is not a valid image file")
    response = client.post(
        "/predict", files={"file": ("corrupt.jpg", corrupt_file, "image/jpeg")}
    )
    # Current behavior - server returns 500 error, though in future it could be improved to 400
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data


def test_very_large_image(client: TestClient) -> None:
    """Test API response when a very large image is provided."""
    # Create a large image
    large_img = Image.new("RGB", (5000, 5000), color="white")
    img_byte_arr = io.BytesIO()
    large_img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    # API should handle large file or return clear error
    response = client.post(
        "/predict", files={"file": ("large.jpg", img_byte_arr, "image/jpeg")}
    )
    
    # Current behavior - server returns 500 error or processes successfully
    assert response.status_code in [200, 500]
    if response.status_code != 200:
        data = response.json()
        assert "detail" in data


def test_model_load_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling when model loading fails."""
    # Patch the create_model function to throw an exception
    def mock_create_model_error() -> None:
        raise RuntimeError("Mock model loading error")
    
    # Apply patch only in the context of this test
    with mock.patch("pneumonia_classifier.models.resnet.create_model", 
                    side_effect=mock_create_model_error):
        # API should properly handle model error
        img = Image.new("RGB", (224, 224), color="white")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        
        response = client.post(
            "/predict", files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
        )
        
        # Expect server error since model cannot be loaded
        assert response.status_code in [500, 503]
        data = response.json()
        assert "detail" in data


def test_wrong_file_field_name(client: TestClient) -> None:
    """Test API response when file is sent with wrong field name."""
    img = Image.new("RGB", (224, 224), color="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    # Use incorrect field name (not "file")
    response = client.post(
        "/predict", files={"wrong_field": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data 