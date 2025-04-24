"""Configure pytest environment."""

# Import hook to prevent CUDA library loading before environment variables are set
import sys
import builtins
import os

# Save the original import function
original_import = builtins.__import__

def import_hook(name, *args, **kwargs):
    """Hook to intercept imports and prevent CUDA libraries from being loaded.
    
    This needs to be set before any torch imports to avoid CUDA initialization.
    """
    if name == 'torch' or name.startswith('torch.'):
        # Set environment variables before importing torch
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CUDA"] = "0"
        os.environ["USE_CUDA"] = "0"
        # Important setting to prevent searching for CUDA libraries
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
        # Disable CUDA warnings
        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

    # Call the original import
    module = original_import(name, *args, **kwargs)
    return module

# Replace builtin import with our hook
builtins.__import__ = import_hook

import pytest
from typing import Generator, Any
from unittest import mock
import torch


# Disable CUDA before importing PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CUDA"] = "0"
os.environ["USE_CUDA"] = "0"

# Disabling CUDA warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


@pytest.fixture(scope="session", autouse=True)
def disable_cuda() -> Generator[None, None, None]:
    """Disable CUDA for all tests."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        # Also explicitly set CPU as default device
        torch.set_default_tensor_type('torch.FloatTensor')
        yield


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """Return CPU device for tests."""
    return torch.device("cpu")


@pytest.fixture
def get_test_image_bytes() -> bytes:
    """Generate a test image as bytes."""
    from PIL import Image
    import io
    
    # Create a simple test image
    img = Image.new("RGB", (224, 224), color="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    
    return img_byte_arr.getvalue() 