"""Configure pytest environment."""

import os
import pytest
from typing import Generator
from unittest import mock


# Disable CUDA before importing PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disabling CUDA warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


@pytest.fixture(scope="session", autouse=True)
def disable_cuda() -> Generator[None, None, None]:
    """Disable CUDA for all tests."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        yield 