"""API routes for pneumonia classification."""

import io
from typing import Any, Dict, Optional, Union, cast

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

# Using module import instead of a specific variable
import pneumonia_classifier.api.main as api_main
from pneumonia_classifier.models.resnet import predict

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint to verify API is working.

    Returns:
        Dict[str, str]: Status message
    """
    return {"status": "healthy"}


@router.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Process X-ray image and return pneumonia prediction results.

    Args:
        file (UploadFile): Uploaded X-ray image

    Returns:
        Dict[str, Any]: Dictionary with prediction results

    Raises:
        HTTPException: If file is not an image or model is not loaded
    """
    # Check if the model is loaded
    if api_main.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Check if the file is an image
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Process image and get prediction
        result = predict(image, model=api_main.model)
        
        # Add file metadata with безопасным приведением типов
        result["filename"] = "" if file.filename is None else file.filename
        result["content_type"] = "" if file.content_type is None else file.content_type
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )
