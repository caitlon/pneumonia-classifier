"""
Dataset and data processing utilities for pneumonia chest X-ray classification.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pneumonia_classifier.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_WORKERS,
)


class ChestXRayDataset(Dataset):
    """Dataset class for chest X-ray images."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        split: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory with dataset
            transform: Image transformations to apply
            split: Data split ('train', 'val', or 'test')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Path to the split directory
        split_dir = self.data_dir / split
        
        # Get all image paths and labels
        self.images: List[str] = []
        self.labels: List[int] = []
        self.class_names: List[str] = []
        
        # Get class folders
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        self.class_names = [d.name for d in class_dirs]
        
        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        
        # Collect images and labels
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in the class directory
            for img_path in class_dir.glob("*.jpeg"):
                self.images.append(str(img_path))
                self.labels.append(class_idx)
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an item by index.
        
        Args:
            idx: Index of the item to fetch
            
        Returns:
            Tuple of (image, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms if specified
        if self.transform:
            image_tensor = self.transform(image)
            return image_tensor, label
        
        # If no transform, create a basic transformation
        default_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        return default_transform(image), label


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Get image transformations for the specified split.
    
    Args:
        split: Data split ('train', 'val', or 'test')
        
    Returns:
        Image transformation pipeline
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])


def create_dataloaders(
    data_dir: Union[str, Path] = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation and testing.
    
    Args:
        data_dir: Path to the data directory
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        Dictionary with dataloaders for train, val, and test splits
    """
    dataloaders: Dict[str, DataLoader] = {}
    
    for split in ["train", "val", "test"]:
        # Create dataset with appropriate transforms
        dataset = ChestXRayDataset(
            data_dir=data_dir,
            transform=get_transforms(split),
            split=split
        )
        
        # Create dataloader
        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    return dataloaders


def get_class_weights(data_dir: Union[str, Path] = DATA_DIR) -> torch.Tensor:
    """
    Calculate class weights based on class distribution to handle imbalance.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Tensor of class weights
    """
    # Get train dataset
    train_dataset = ChestXRayDataset(
        data_dir=data_dir,
        transform=None,
        split="train"
    )
    
    # Get class counts
    class_counts = torch.zeros(len(train_dataset.class_names))
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        class_counts[label] += 1
    
    # Calculate weights (inverse of frequency)
    weights = 1.0 / class_counts
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    # Explicitly specify the type
    weights_tensor: torch.Tensor = weights
    
    return weights_tensor


def process_image(image: Image.Image) -> torch.Tensor:
    """
    Processes an image for prediction.
    
    Args:
        image: PIL image
        
    Returns:
        Image tensor for the model
    """
    # Apply validation transformations and add batch dimension
    transform = get_transforms("val")
    result: torch.Tensor = transform(image).unsqueeze(0)
    return result 