"""FastAPI application for pneumonia classification."""

import os
if "PYTEST_CURRENT_TEST" in os.environ or "CI" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from typing import Dict, Optional

import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pneumonia_classifier.api.routes import router
from pneumonia_classifier.config import API_CONFIG, TRAINING_CONFIG
from pneumonia_classifier.models.resnet import load_model
from pneumonia_classifier.utils import get_device

# Global variable for storing the model
model: Optional[nn.Module] = None

# Initialize FastAPI application
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
)

# Add CORS middleware for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect routes
app.include_router(router)


@app.on_event("startup")
async def startup_event() -> None:
    """Loads the model when the application starts."""
    global model
    model_path = os.environ.get("MODEL_PATH", TRAINING_CONFIG["default_model_path"])

    try:
        device = get_device()
        model = load_model(model_path, device)
        print(f"Model successfully loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "API for pneumonia classification using X-ray images",
        "usage": ("Send a POST request with an image to /predict"),
    }
