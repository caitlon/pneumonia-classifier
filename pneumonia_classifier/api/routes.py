"""API routes for pneumonia classification."""

import io
from typing import Dict, List, Any
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException
import torch

# Using module import instead of a specific variable
import pneumonia_classifier.api.main as api_main
from pneumonia_classifier.utils import process_image, get_device

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint to verify API is working.
    
    Returns:
        Dict[str, str]: Status message
    """
    return {"status": "ok"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
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
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process image for model input
        processed_image = process_image(image)
        
        # Get prediction
        device = get_device()
        with torch.no_grad():
            outputs = api_main.model(processed_image.to(device))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        class_names = ["Normal", "Pneumonia"]
        probabilities_list = probabilities.cpu().numpy().tolist()
        
        return {
            "predicted_class": class_names[predicted_class],
            "confidence": round(float(probabilities[predicted_class]) * 100, 2),
            "probabilities": {
                class_name: round(float(prob) * 100, 2) 
                for class_name, prob in zip(class_names, probabilities_list)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}") 